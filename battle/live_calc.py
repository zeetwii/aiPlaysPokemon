"""Interactive damage-calculator bench, driven by the live emulator.

Play the game in mGBA (with mgba_server.lua loaded) and run this alongside it.
It pulls the real battle state over the socket, builds `damage_calc.Pokemon`
objects out of it and shows what the calculator predicts for both sides.

    python battle/live_calc.py            # REPL
    python battle/live_calc.py watch      # start straight in watch mode
    python battle/live_calc.py --log runs.jsonl

The point of `watch` is verification, not display: it polls the battle, and
every time a HP bar moves it checks the drop against the rolls it predicted
from the *previous* snapshot. A hit that lands outside every predicted range
is a bug in the calculator (or a modifier it does not model yet), and gets
reported as such along with a running tally.

Caveats worth knowing when reading a mismatch:
  * gBattleMons stats are pre-stage; stages are applied by the calculator.
    If the game ever hands us already-modified stats, boosted mons will read
    high by exactly the stage multiplier.
  * Multi-hit moves, recoil, held-berry recovery and residual damage all move
    HP without a single matching roll. Residuals are guessed at and labelled.
  * Weather, screens and Helping Hand are not in the game state we read --
    set them by hand with `set` if a fight involves them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field as dc_field
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent / "mGBA"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from damage_calc import (  # noqa: E402
    Context,
    DataError,
    Field,
    GameData,
    Move,
    Pokemon,
    accuracy_chance,
    calculate,
    critical_hit_rate,
    expected_damage,
)
from mgba_client import MGBAClient, MGBAError  # noqa: E402

# The server reports status as a 3-letter code; the calculator wants the
# lowercase names it checks for (burn/paralysis/... ).
STATUS_MAP = {
    "OK": "", "SLP": "sleep", "PSN": "poison", "BRN": "burn",
    "FRZ": "freeze", "PAR": "paralysis", "TOX": "toxic",
}

STAGE_MAP = {
    "attack": "atk", "defense": "def", "speed": "spe",
    "sp_attack": "spa", "sp_defense": "spd",
    "accuracy": "acc", "evasion": "eva",
}

WEATHERS = ("", "rain", "sun", "sandstorm", "hail")

BAR_WIDTH = 20


# --------------------------------------------------------------------------
# State -> calculator objects
# --------------------------------------------------------------------------


def _clean(value: str | None) -> str:
    """Blank out the server's placeholder strings."""
    if not value or value in ("None", "Unknown", "???"):
        return ""
    return value


def _types(mon: dict) -> tuple[str, ...]:
    """Deduplicated type tuple.

    Deduplication matters: the calculator walks the defender's types one at a
    time, so a mon reported as Normal/Normal would take the multiplier twice.
    """
    out = []
    for key in ("type1", "type2"):
        t = _clean(mon.get(key))
        if t and t not in out:
            out.append(t)
    return tuple(out)


def to_pokemon(mon: dict) -> Pokemon:
    """Build a calculator Pokemon from a battle mon or a party mon dict."""
    stages = {}
    for game_key, calc_key in STAGE_MAP.items():
        stages[calc_key] = int(mon.get("stat_stages", {}).get(game_key, 0))

    return Pokemon(
        species=mon.get("species", "?"),
        level=int(mon.get("level", 1)),
        types=_types(mon),
        stats={
            "atk": int(mon.get("attack", 1)),
            "def": int(mon.get("defense", 1)),
            "spa": int(mon.get("sp_attack", 1)),
            "spd": int(mon.get("sp_defense", 1)),
            "spe": int(mon.get("speed", 1)),
        },
        max_hp=max(1, int(mon.get("max_hp", 1))),
        current_hp=int(mon.get("hp", 0)),
        ability=_clean(mon.get("ability")),
        item=_clean(mon.get("held_item")),
        status=STATUS_MAP.get(mon.get("status", "OK"), ""),
        stages=stages,
        friendship=int(mon.get("friendship", 70)),
    )


def move_names(mon: dict) -> list[str]:
    return [m["name"] for m in mon.get("moves", []) if m.get("name")]


def pp_of(mon: dict, move_name: str) -> int | None:
    for m in mon.get("moves", []):
        if m.get("name") == move_name:
            return m.get("pp")
    return None


@dataclass
class Snapshot:
    """One poll of the game, reshaped into what the calculator needs."""

    raw: dict
    in_battle: bool
    you: Pokemon | None = None
    foe: Pokemon | None = None
    you_raw: dict = dc_field(default_factory=dict)
    foe_raw: dict = dc_field(default_factory=dict)
    taken_at: float = 0.0

    @classmethod
    def capture(cls, client: MGBAClient) -> "Snapshot":
        raw = client.game_state()
        snap = cls(raw=raw, in_battle=bool(raw.get("in_battle")), taken_at=time.time())
        if not snap.in_battle:
            return snap

        battle = raw.get("battle") or {}
        you_raw = battle.get("player_active")
        foe_raw = battle.get("enemy_active")

        # gBattleMons is not populated for the first few frames of a battle;
        # the party structs are, so fall back to them rather than showing
        # nothing at all during the intro animation.
        if not you_raw:
            party = raw.get("party") or []
            you_raw = next((p for p in party if p.get("hp", 0) > 0), None)
        if not foe_raw:
            enemy = raw.get("enemy_party") or []
            foe_raw = next((p for p in enemy if p.get("hp", 0) > 0), None)

        if you_raw:
            snap.you_raw = you_raw
            snap.you = to_pokemon(you_raw)
        if foe_raw:
            snap.foe_raw = foe_raw
            snap.foe = to_pokemon(foe_raw)
        return snap

    @property
    def ready(self) -> bool:
        return self.you is not None and self.foe is not None

    def identity(self) -> tuple:
        """Changes when either side switches (used to reset HP tracking)."""
        return (
            self.you_raw.get("species"), self.you_raw.get("nickname"),
            self.you_raw.get("level"), self.you_raw.get("max_hp"),
            self.foe_raw.get("species"), self.foe_raw.get("nickname"),
            self.foe_raw.get("level"), self.foe_raw.get("max_hp"),
        )


# --------------------------------------------------------------------------
# Session: the toggles the game state cannot tell us about
# --------------------------------------------------------------------------


@dataclass
class Session:
    data: GameData
    field: Field = dc_field(default_factory=Field)
    ctx: Context = dc_field(default_factory=Context)
    checks: int = 0
    matches: int = 0
    log_path: Path | None = None

    def resolve(self, name: str) -> Move | None:
        try:
            return self.data.move(name)
        except DataError:
            return None

    def log(self, record: dict):
        if not self.log_path:
            return
        record["time"] = datetime.now().isoformat(timespec="seconds")
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def hp_bar(current: int, maximum: int) -> str:
    if maximum <= 0:
        return "?" * BAR_WIDTH
    filled = max(0, min(BAR_WIDTH, round(BAR_WIDTH * current / maximum)))
    return "#" * filled + "." * (BAR_WIDTH - filled)


def effective_speed(mon: Pokemon) -> int:
    speed = mon.stat("spe")
    if mon.status == "paralysis":
        speed //= 4                      # Gen 3 paralysis is a flat quarter
    if mon.item.lower() in ("macho brace", "iron ball"):
        speed //= 2
    if mon.ability.lower() == "chlorophyll":
        pass                             # weather-dependent; left to the user
    return max(1, speed)


def describe(mon: Pokemon, raw: dict) -> str:
    nick = raw.get("nickname") or mon.species
    label = mon.species if nick.upper() == mon.species.upper() else f"{nick} ({mon.species})"
    bits = [f"Lv{mon.level} {label}", "/".join(mon.types) or "?"]
    if mon.ability:
        bits.append(mon.ability)
    if mon.item:
        bits.append(f"@{mon.item}")
    if mon.status:
        bits.append(mon.status.upper())
    stages = {k: v for k, v in mon.stages.items() if v}
    if stages:
        bits.append(" ".join(f"{k}{v:+d}" for k, v in sorted(stages.items())))
    return "  ".join(bits)


def hp_line(mon: Pokemon) -> str:
    pct = 100.0 * mon.current_hp / mon.max_hp if mon.max_hp else 0.0
    return f"[{hp_bar(mon.current_hp, mon.max_hp)}] {mon.current_hp}/{mon.max_hp} ({pct:.0f}%)"


@dataclass
class Row:
    move: Move
    result: object                      # DamageResult
    crit: object
    expected: float
    accuracy: float
    crit_rate: float
    pp: int | None

    @property
    def is_damaging(self) -> bool:
        return not self.move.is_status and self.result.max_damage > 0

    @property
    def is_usable(self) -> bool:
        """Can this move be picked at all this turn?

        A move at zero PP is not a worse option than the others, it is not an
        option: the game refuses it. Unknown PP (None) reads as usable on
        purpose - the same bargain PPWatcher makes - because withholding a move
        the calculator merely failed to read a number for is the expensive way
        to be wrong.
        """
        return self.pp is None or self.pp > 0


def build_rows(session: Session, attacker: Pokemon, defender: Pokemon,
               names, raw_mon: dict) -> list[Row]:
    rows = []
    for name in names:
        move = session.resolve(name)
        if move is None:
            continue
        normal_ctx = Context(**{**session.ctx.__dict__, "critical": False})
        crit_ctx = Context(**{**session.ctx.__dict__, "critical": True})
        rows.append(Row(
            move=move,
            result=calculate(session.data, attacker, defender, move, session.field, normal_ctx),
            crit=calculate(session.data, attacker, defender, move, session.field, crit_ctx),
            expected=expected_damage(session.data, attacker, defender, move,
                                     session.field, normal_ctx),
            accuracy=accuracy_chance(move, attacker, defender),
            crit_rate=critical_hit_rate(move, attacker),
            pp=pp_of(raw_mon, name),
        ))
    rows.sort(key=lambda r: (r.expected, r.is_damaging), reverse=True)
    return rows


def ko_text(row: Row, defender: Pokemon) -> str:
    res = row.result
    if not row.is_damaging:
        return "-"
    if res.guaranteed_ko:
        return "OHKO"
    if res.possible_ko:
        return f"OHKO {res.ko_chance:.0%}"
    avg = row.expected or res.average_damage
    if avg <= 0:
        return "-"
    return f"{-(-defender.current_hp // max(1, int(avg)))}HKO"


def unknown_move_names(session: Session, names) -> list[str]:
    return [n for n in names if session.resolve(n) is None]


def print_rows(session: Session, rows: list[Row], defender: Pokemon, unknown: list[str]):
    print(f"     {'MOVE':<13} {'TYPE':<9}{'CAT':<5}{'PP':>3}  {'ACC':>4}  "
          f"{'DAMAGE':<11} {'% OF HP':<11} {'CRIT':<9} KO")
    for i, row in enumerate(rows, 1):
        res = row.result
        if row.move.is_status:
            effect = (row.move.effect or "status move").split(".")[0].strip()
            if len(effect) > 52:
                effect = effect[:49].rstrip() + "..."
            print(f" {i:>2}  {row.move.name:<13} {row.move.type:<9}{'Stat':<5}"
                  f"{(row.pp if row.pp is not None else '-'):>3}  "
                  f"{int(row.accuracy * 100):>3}%  {effect}")
            if not row.is_usable:
                print(f"     {'':<13} ! out of PP - the game will not let you "
                      f"pick this move")
            continue

        dmg = f"{res.min_damage}-{res.max_damage}"
        pct = (f"{100.0 * res.min_damage / defender.max_hp:.0f}-"
               f"{100.0 * res.max_damage / defender.max_hp:.0f}%")
        crit = f"{row.crit.min_damage}-{row.crit.max_damage}"
        eff = {0: "x0", 0.25: "x1/4", 0.5: "x1/2", 1.0: "", 2.0: "x2", 4.0: "x4"}.get(
            res.effectiveness, f"x{res.effectiveness:g}")
        tag = " ".join(t for t in ("STAB" if res.stab else "", eff) if t)
        print((f" {i:>2}  {row.move.name:<13} {row.move.type:<9}"
               f"{('Phys' if res.category == 'Physical' else 'Spec'):<5}"
               f"{(row.pp if row.pp is not None else '-'):>3}  "
               f"{int(row.accuracy * 100):>3}%  {dmg:<11} {pct:<11} {crit:<9} "
               f"{ko_text(row, defender):<8} exp {row.expected:.1f}  {tag}").rstrip())
        if not row.is_usable:
            print(f"     {'':<13} ! out of PP - the game will not let you pick "
                  f"this move")
        for note in res.notes:
            print(f"     {'':<13} ! {note}")
    if unknown:
        print(f"     (not in moves.json: {', '.join(unknown)})")


def print_matchup(session: Session, snap: Snapshot):
    if not snap.in_battle:
        print("Not in battle.")
        party = snap.raw.get("party") or []
        for i, pk in enumerate(party, 1):
            print(f"  {i}. Lv{pk['level']:>3} {pk['species']:<12} "
                  f"HP {pk['hp']}/{pk['max_hp']}")
        return
    if not snap.ready:
        print("In battle, but the battler structs are not populated yet.")
        return

    you, foe = snap.you, snap.foe
    print()
    print("=" * 78)
    print(f"YOU  {describe(you, snap.you_raw)}")
    print(f"     {hp_line(you)}")
    print(f"FOE  {describe(foe, snap.foe_raw)}")
    print(f"     {hp_line(foe)}")

    ys, fs = effective_speed(you), effective_speed(foe)
    order = "you move first" if ys > fs else ("foe moves first" if fs > ys else "speed tie")
    print(f"     Speed {ys} vs {fs} -- {order}")

    active = active_modifiers(session)
    if active:
        print(f"     Modifiers: {active}")
    print("-" * 78)

    names = move_names(snap.you_raw)
    print_rows(session, build_rows(session, you, foe, names, snap.you_raw), foe,
               unknown_move_names(session, names))

    print("-" * 78)
    print("INCOMING")
    fnames = move_names(snap.foe_raw)
    print_rows(session, build_rows(session, foe, you, fnames, snap.foe_raw), you,
               unknown_move_names(session, fnames))
    print("=" * 78)


def active_modifiers(session: Session) -> str:
    bits = []
    if session.field.weather:
        bits.append(session.field.weather)
    if session.field.reflect:
        bits.append("reflect")
    if session.field.light_screen:
        bits.append("light screen")
    if session.field.is_double:
        bits.append("doubles")
    if session.ctx.helping_hand:
        bits.append("helping hand")
    if session.ctx.critical:
        bits.append("FORCED CRIT")
    return ", ".join(bits)


def print_detail(session: Session, snap: Snapshot, index_or_name: str):
    """Full breakdown for one move: inputs, every roll, and the crit variant."""
    if not snap.ready:
        print("Not in a battle with both battlers loaded.")
        return
    names = move_names(snap.you_raw)
    rows = build_rows(session, snap.you, snap.foe, names, snap.you_raw)

    row = None
    if index_or_name.isdigit() and 1 <= int(index_or_name) <= len(rows):
        row = rows[int(index_or_name) - 1]
    else:
        move = session.resolve(index_or_name)
        if move is None:
            print(f"Unknown move: {index_or_name}")
            return
        row = next((r for r in rows if r.move.key == move.key), None)
        if row is None:
            rows2 = build_rows(session, snap.you, snap.foe, [move.name], snap.you_raw)
            row = rows2[0] if rows2 else None
    if row is None:
        print("No such move.")
        return

    res, crit = row.result, row.crit
    print()
    print(f"{row.move.name}  ({row.move.type}, {res.category}, "
          f"base {row.move.power or '-'}, acc {row.move.accuracy or '-'})")
    print(f"  effect:       {row.move.effect}")
    print(f"  attacker:     {snap.you.species} Lv{snap.you.level}  "
          f"A={res.attack}  effective power={res.power}")
    print(f"  defender:     {snap.foe.species} Lv{snap.foe.level}  "
          f"D={res.defense}  HP={snap.foe.current_hp}/{snap.foe.max_hp}")
    print(f"  STAB:         {res.stab}    effectiveness: x{res.effectiveness:g}")
    print(f"  accuracy:     {row.accuracy:.1%}    crit rate: {row.crit_rate:.1%}")
    print(f"  rolls:        {list(res.rolls)}")
    print(f"  crit rolls:   {list(crit.rolls)}")
    print(f"  expected/turn {row.expected:.2f}  (accuracy- and crit-weighted)")
    print(f"  summary:      {res.summary()}")
    if res.notes:
        for note in res.notes:
            print(f"  note:         {note}")


# --------------------------------------------------------------------------
# Watch mode: predict, then check the prediction against the real HP change
# --------------------------------------------------------------------------


def _candidates(session: Session, attacker: Pokemon, defender: Pokemon,
                names, raw_mon: dict, delta: int) -> list[str]:
    """Which of `names` could have produced a `delta` HP drop."""
    hits = []
    for row in build_rows(session, attacker, defender, names, raw_mon):
        if row.move.is_status:
            continue
        if delta in row.result.rolls:
            hits.append(f"{row.move.name} ({row.result.min_damage}-{row.result.max_damage})")
        elif delta in row.crit.rolls:
            hits.append(f"{row.move.name} CRIT ({row.crit.min_damage}-{row.crit.max_damage})")
        elif delta >= defender.current_hp:
            # A KO truncates the drop at the remaining HP, so the roll that
            # killed is unknowable -- any move that could reach counts.
            if row.result.max_damage >= defender.current_hp:
                hits.append(f"{row.move.name} (KO, >= {defender.current_hp})")
            elif row.crit.max_damage >= defender.current_hp:
                hits.append(f"{row.move.name} CRIT (KO, >= {defender.current_hp})")
    return hits


def _residual_guess(defender: Pokemon, delta: int) -> str | None:
    eighth, sixteenth = max(1, defender.max_hp // 8), max(1, defender.max_hp // 16)
    if delta == eighth:
        return "1/8 max HP -- burn/poison/leech seed?"
    if delta == sixteenth:
        return "1/16 max HP -- sandstorm/hail/toxic tick?"
    return None


def check_damage(session: Session, prev: Snapshot, curr: Snapshot) -> list[str]:
    """Compare the HP deltas between two snapshots against the predictions."""
    lines = []
    pairs = (
        ("FOE", prev.foe, curr.foe, prev.you, move_names(prev.you_raw), prev.you_raw),
        ("YOU", prev.you, curr.you, prev.foe, move_names(prev.foe_raw), prev.foe_raw),
    )
    for label, before, after, attacker, names, raw_mon in pairs:
        if before is None or after is None:
            continue
        delta = before.current_hp - after.current_hp
        if delta == 0:
            continue
        if delta < 0:
            lines.append(f"  [heal] {label} {before.species} +{-delta} HP "
                         f"-> {after.current_hp}/{after.max_hp}")
            continue

        hits = _candidates(session, attacker, before, names, raw_mon, delta)
        session.checks += 1
        if hits:
            session.matches += 1
            verdict = "OK  " + " | ".join(hits)
        else:
            residual = _residual_guess(before, delta)
            verdict = ("?   " + residual) if residual else "MISS  no predicted roll matches"
        lines.append(f"  [hit ] {label} {before.species} -{delta} HP "
                     f"-> {after.current_hp}/{after.max_hp}   {verdict}")
        if not hits and not _residual_guess(before, delta):
            for row in build_rows(session, attacker, before, names, raw_mon):
                if not row.move.is_status:
                    lines.append(f"          predicted {row.move.name}: "
                                 f"{row.result.min_damage}-{row.result.max_damage} "
                                 f"(crit {row.crit.min_damage}-{row.crit.max_damage})")
        session.log({
            "event": "hit", "side": label, "delta": delta,
            "attacker": attacker.species, "defender": before.species,
            "defender_hp_before": before.current_hp,
            "matched": bool(hits), "candidates": hits,
        })
    return lines


def watch(session: Session, client: MGBAClient, interval: float = 0.4):
    """Poll the battle, reprinting on a switch and checking every HP change."""
    print("Watching -- Ctrl-C to return to the prompt.")
    prev: Snapshot | None = None
    last_identity = None
    was_in_battle = None
    try:
        while True:
            try:
                curr = Snapshot.capture(client)
            except (MGBAError, ConnectionError, OSError) as exc:
                print(f"  (poll failed: {exc})")
                time.sleep(1.0)
                continue

            if curr.in_battle != was_in_battle:
                print("\n*** BATTLE START ***" if curr.in_battle else "\n*** BATTLE END ***")
                was_in_battle = curr.in_battle
                prev = None
                last_identity = None

            if curr.in_battle and curr.ready:
                identity = curr.identity()
                if identity != last_identity:
                    print_matchup(session, curr)
                    last_identity = identity
                    prev = curr
                else:
                    if prev is not None:
                        for line in check_damage(session, prev, curr):
                            print(line)
                    prev = curr
            time.sleep(interval)
    except KeyboardInterrupt:
        print(f"\nStopped. Checks: {session.checks}, matched: {session.matches}")


# --------------------------------------------------------------------------
# REPL
# --------------------------------------------------------------------------

HELP = """Commands:
  <enter> / r        re-read the game and print the matchup
  w                  watch mode: auto-refresh and verify every HP change
  d <n|move>         full breakdown for one of your moves
  calc <move>        run an arbitrary move from your active mon
  incoming <move>    run an arbitrary move from the foe against you
  set <key> <value>  weather rain|sun|sandstorm|hail|none, reflect on|off,
                     lightscreen on|off, doubles on|off, helpinghand on|off,
                     crit on|off
  show               current modifiers and the verification tally
  hp <you|foe> <n>   override current HP for a what-if calculation
  state              pretty-print the raw game state
  tap <btn> [frames] press a button in the emulator
  q                  quit
"""


def cmd_set(session: Session, args: list[str]):
    if len(args) < 2:
        print("Usage: set <key> <value>")
        return
    key, value = args[0].lower(), args[1].lower()
    on = value in ("on", "true", "1", "yes")
    if key == "weather":
        value = "" if value in ("none", "off", "clear") else value
        if value not in WEATHERS:
            print(f"Weather must be one of: none, {', '.join(w for w in WEATHERS if w)}")
            return
        session.field.weather = value
    elif key == "reflect":
        session.field.reflect = on
    elif key in ("lightscreen", "light_screen"):
        session.field.light_screen = on
    elif key in ("doubles", "double"):
        session.field.is_double = on
    elif key in ("helpinghand", "helping_hand"):
        session.ctx.helping_hand = on
    elif key == "crit":
        session.ctx.critical = on
    else:
        print(f"Unknown key: {key}")
        return
    print(f"Modifiers: {active_modifiers(session) or '(none)'}")


def cmd_calc(session: Session, snap: Snapshot, name: str, incoming: bool):
    if not snap.ready:
        print("Not in a battle with both battlers loaded.")
        return
    move = session.resolve(name)
    if move is None:
        print(f"Unknown move: {name}")
        return
    attacker, defender, raw = (
        (snap.foe, snap.you, snap.foe_raw) if incoming else (snap.you, snap.foe, snap.you_raw))
    rows = build_rows(session, attacker, defender, [move.name], raw)
    print_rows(session, rows, defender, [])


def cmd_hp(snap: Snapshot, args: list[str]):
    if len(args) < 2 or not snap.ready:
        print("Usage: hp <you|foe> <value>   (requires an active battle)")
        return
    target = snap.you if args[0].lower().startswith("y") else snap.foe
    try:
        target.current_hp = max(0, min(target.max_hp, int(args[1])))
    except ValueError:
        print("HP must be a number.")
        return
    print(f"{target.species} HP set to {target.current_hp}/{target.max_hp} "
          "(overrides until the next refresh)")


def repl(session: Session, client: MGBAClient):
    snap = Snapshot.capture(client)
    print_matchup(session, snap)
    print("\nType 'help' for commands.")

    while True:
        try:
            line = input("calc> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        parts = line.split()
        cmd = parts[0].lower() if parts else "r"
        args = parts[1:]

        try:
            if cmd in ("q", "quit", "exit"):
                break
            elif cmd in ("help", "?", "h"):
                print(HELP)
            elif cmd in ("r", "refresh"):
                snap = Snapshot.capture(client)
                print_matchup(session, snap)
            elif cmd in ("w", "watch"):
                watch(session, client)
                snap = Snapshot.capture(client)
            elif cmd in ("d", "detail"):
                if not args:
                    print("Usage: d <n|move>")
                else:
                    print_detail(session, snap, " ".join(args))
            elif cmd == "calc":
                cmd_calc(session, snap, " ".join(args), incoming=False)
            elif cmd == "incoming":
                cmd_calc(session, snap, " ".join(args), incoming=True)
            elif cmd == "set":
                cmd_set(session, args)
            elif cmd == "show":
                print(f"Modifiers: {active_modifiers(session) or '(none)'}")
                print(f"Verified hits: {session.matches}/{session.checks}"
                      + (f"   log: {session.log_path}" if session.log_path else ""))
            elif cmd == "hp":
                cmd_hp(snap, args)
            elif cmd == "state":
                from mgba_client import print_game_state
                print_game_state(Snapshot.capture(client).raw)
            elif cmd == "tap":
                if not args:
                    print("Usage: tap <button> [frames]")
                else:
                    client.tap(args[0], int(args[1]) if len(args) > 1 else None)
                    snap = Snapshot.capture(client)
            else:
                print(f"Unknown command: {cmd}  (try 'help')")
        except (MGBAError, ValueError, DataError) as exc:
            print(f"Error: {exc}")
        except (ConnectionError, OSError) as exc:
            print(f"Connection lost: {exc}")
            break

    print(f"Verified hits: {session.matches}/{session.checks}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", nargs="?", choices=("repl", "watch"), default="repl")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=54321)
    parser.add_argument("--interval", type=float, default=0.4,
                        help="watch-mode poll interval in seconds")
    parser.add_argument("--log", type=Path, default=None,
                        help="append every verified hit to this JSONL file")
    args = parser.parse_args()

    session = Session(data=GameData.load(), log_path=args.log)

    try:
        client = MGBAClient(host=args.host, port=args.port)
    except OSError as exc:
        print(f"Could not connect to mGBA on {args.host}:{args.port} -- {exc}")
        print("Load mgba_server.lua in mGBA (Tools > Scripting) and try again.")
        return 1

    with client:
        print(f"Connected to {args.host}:{args.port}")
        if args.mode == "watch":
            watch(session, client, args.interval)
        else:
            repl(session, client)
    return 0


if __name__ == "__main__":
    sys.exit(main())
