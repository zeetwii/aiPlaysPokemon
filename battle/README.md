# Battle Tools

Battles in Pokemon are another great example of why you would create a tool for a model.  While there are numerous stratigies, esspically in competitive play, battles themselves are basically giant math formulas.  Rather than asking a model to try and generate or recall this every turn, it's far more memory efiecent to move as much as possible into external files and programs.  For example, [pokedex](./pokedex.json) contains the Generation III pokedex, [typeChart](./typeChart.json) contains the Generation III type chart, and [moves](./moves.json) contains every move in Fire Red and Leaf Green, and their effects.  [moveData](./moveData.json) and [learnsets](./learnsets.json) are the same sort of thing pulled straight from the ROM - see "Getting the Move Data" below.  

## Getting Data

Like with [location tracking](../locationTracking/README.md), the original goal was to try and use Optical Character Recognition(OCR) to do detections and processing of how the battle waas going.  This quickly failed, the gameboy advance has a very small screen, and thus all the games on it play natively at a small resolution.  While the game is pretty readable by human standards, OCR models would get confused over relatively import details, like HP or the name of the opposing Pokemon.  In theory this could have been fixed by pulling out all the alphabet sprites, and training an OCR model on those, but for DEFCON the faster and easier thing to do was to just fall back on using game state.  

Getting the data from the game state of the emulator actually solves a lot of problems down the line, since the game knows not only if we are fighting and if so what monster, but also all of the opponents stats.  This gives us a very accurate damage calculation, which is about the only win we get.  Movesets in the early Pokemon games were not great, esspically for your starters.  Most really powerful moves are gated behind a TM, and as of a week before DEFCON, I plan to leave item management to the AI as a hopefully ammusing comparision to all of the tools being designed for the rest of the game.  So who knows if anything will ever learn hyper beam.  As such, we actually need the battle AI to be pretty smart, and able to plan around the use of status effect moves in addtion to damaging ones.

## Damage Calculation

There are a lot of Pokemon damage calculators tailored made for everything from nuzlocks to competitive play.  However, because the goal of this project was to run live at DEFCON, where WiFi is fickle at best, I wanted to make sure the AI player had their own local damage calculator to use.  For this initial pass it's just doing the basics of how many turns will it take for move X to knock out Y, vs how soon can they knock out your own pokemon.  Eventually I want to better represent stalling tactics, like starting a long fight with leech seed to recover HP over multiple turns, or using sleep powder to stall a stronger opponent, but that will likely happen after the conference.  


### Damage Formula

TODO:  Grab the Gen III Damage calc and explain it here along with how we get stuff from game state

## Battle Planning

The other nice thing about having the calculator as a local script is we can build around it for being able to tell the player AI when they are actually strong enough to likely win against the next gym battle or other important encounter.  [Trainers.json](./trainers.json) has a list of pulled trainers from the game that represent key encounters, while [matchup](./matchup.py) handles calculating how good or bad the current players team does against these extracted encounters.  This then gives us a way to inform the player AI that they need to go train more by either fighting trainers or wild pokemon.  

## Learning Moves

[move_learn](./move_learn.py) handles the level-up prompt, which turned out to be the single worst moment in the game for the AI. It is the only decision in the run that cannot be undone, and until this existed the harness could not see it at all: the game keeps `in_battle` set the whole time, so the report cheerfully printed a damage table and offered `use`/`switch`/`bag`/`run` while the screen was actually a five-row move list waiting on a cursor. Every one of those commands is four or five button presses, and those presses went into the list. Bulbasaur, which learns two status moves at level 15 and its only real attack at 20, would come out of it knowing POISONPOWDER, SLEEP POWDER, GROWL and LEECH SEED - completely unable to hurt anything.

The trick that makes the advice work is ranking **movesets** rather than moves. Asked "which of these five moves is worst", POISONPOWDER against a 35-power Tackle is genuinely arguable, and three defensible answers in a row is a disaster. Asked "which of these five movesets is worst", the disaster is something you can just look at: it has one attacking move in it. So there are hard constraints - keep at least two attacking moves, at least one of them same-type, no more than two status moves - checked before anything is weighed, and then the usual judgement calls (type coverage, expected damage, PP, and how much of a turn a status move actually buys you) decide between what is left.

It also reads the learnset, because the thing you most want to know at level 15 is that RAZOR LEAF is coming at 20 and that the *other* powder is landing at this same level, seconds from now.

Three bits of state make it reliable, all read straight out of RAM rather than guessed from pixels:

* `gMoveToLearn` says which move is being offered. It's sticky - it keeps the last value long after the prompt is gone - so it answers "which move", never "is there a question".
* the move list's cursor is a single byte, which is what lets `forget <move>` aim and verify instead of counting taps. The list wraps in both directions, so dead reckoning would be a guess, and a wrong guess deletes your best attack and reports success.
* battle messages live in their own buffer. `gStringVar4` is not updated during a battle at all, so during a level-up prompt it is still showing whatever the last overworld NPC said.

## Getting the Move Data

The JSON in here was originally typed up from a reference site, which is fine for things a human reads - the effect prose in moves.json is better written than anything in the ROM - and no good for things a program compares. It has no PP, no move ids, and describes effects only as English sentences.

[rom_dump](./rom_dump.py) pulls the real tables out of the cartridge instead: `gMoveNames`, `gBattleMoves` (power, type, accuracy, PP, priority and an exact effect id) and `gLevelUpLearnsets`. Run it once with the emulator open and it writes [moveData](./moveData.json) and [learnsets](./learnsets.json), which everything else then reads offline like the rest of this folder. It verifies the addresses against known values first, so a different ROM fails loudly instead of writing a file full of plausible nonsense.

One thing we still don't do is recommend moves when you gain a new TM. That's the same scoring problem with a different trigger, so it should be a much smaller job now.

