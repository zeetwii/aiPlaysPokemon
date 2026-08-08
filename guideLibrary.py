"""
Read the guide mirror on the player's behalf.

guideScraper.py writes guides/; this is the other half, the part that answers
two questions at the table:

    "I am standing on 3-1-ViridianCity"  -> the steps for Viridian City
    "where do I find Abra"               -> the dex line that says

Both answers are cut to a character budget before they go anywhere near a
prompt. A local model on a laptop has room for a paragraph of guide, not for
the 26KB of Silph Co., so every entry point here takes a budget and honours it.

Nothing in here fails loudly. A missing or half-built guides/ directory just
means `available` is False and every lookup comes back empty, because the
player has played fine without the mirror and should keep doing so.
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Words too common to say anything about which section of a page is the one
# the current objective is talking about.
_NOISE = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'can', 'for',
    'from', 'get', 'go', 'has', 'have', 'in', 'into', 'is', 'it', 'its', 'of',
    'on', 'once', 'one', 'or', 'out', 'that', 'the', 'then', 'there', 'this',
    'to', 'up', 'was', 'what', 'when', 'where', 'which', 'will', 'with', 'you',
    'your', 'pokemon', 'firered', 'leafgreen', 'walkthrough', 'guide',
}


def _words(text):
    return {word for word in re.findall(r"[a-z0-9']+", (text or '').lower())
            if len(word) > 2 and word not in _NOISE}


def _trim(text, budget):
    """Cut to budget on a paragraph break, then a sentence, then a word."""
    text = text.strip()
    if len(text) <= budget:
        return text
    window = text[:budget]
    for boundary in ('\n\n', '. ', ' '):
        cut = window.rfind(boundary)
        if cut > budget // 3:
            return window[:cut].rstrip(' .,;') + ' ...'
    return window.rstrip() + ' ...'


def _plain(line):
    """A Markdown table row or bullet, as something a model reads as a fact."""
    if line.startswith('|'):
        cells = [cell.strip() for cell in line.strip('|').split('|')]
        text = ' — '.join(cell for cell in cells if cell)
    else:
        text = re.sub(r'^[-*\s]+', '', line)
    # Emphasis markers are formatting, and a fact quoted into a prompt should
    # not arrive wearing them.
    return re.sub(r'\*+', '', text).strip()


def _readable(text):
    """Markdown flattened for a prompt: no pipes, no rules, no emphasis.

    The mirror is written to be read as Markdown; a prompt is read as lines.
    A trainer row is worth keeping, its column rule is not.
    """
    # A link is an offer to go and read something else, which a prompt cannot
    # take up. Keep the words, drop the destination.
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)

    lines = []
    for line in text.split('\n'):
        stripped = line.strip()
        if not stripped:
            lines.append('')
        elif set(stripped) <= set('|-: '):        # table rules, horizontal rules
            continue
        elif stripped.startswith('#'):
            lines.append(re.sub(r'^#+\s*', '', stripped))
        elif stripped.startswith('>'):
            lines.append(re.sub(r'^>\s*', '', stripped))
        elif stripped.startswith('|') or stripped.startswith('- '):
            lines.append(_plain(stripped))
        else:
            lines.append(re.sub(r'\*+', '', stripped))
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip()


class GuideLibrary:
    """The mirror, indexed by map id and searchable by name."""

    def __init__(self, root=None):
        self.root = Path(root) if root else HERE / 'guides'
        self.byMapId = {}
        self.reference = []
        self.pages = []
        self._bodies = {}

        try:
            data = json.loads((self.root / 'index.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return
        self.byMapId = data.get('byMapId') or {}
        self.reference = data.get('reference') or []
        self.pages = data.get('pages') or []

    @property
    def available(self):
        return bool(self.byMapId or self.reference)

    # ---- reading -----------------------------------------------------------

    def _body(self, relative):
        """A page's Markdown with its frontmatter removed, read once."""
        if relative not in self._bodies:
            try:
                raw = (self.root / relative).read_text(encoding='utf-8')
            except OSError:
                raw = ''
            if raw.startswith('---\n'):
                end = raw.find('\n---\n', 4)
                raw = raw[end + 5:] if end != -1 else raw
            self._bodies[relative] = raw.strip()
        return self._bodies[relative]

    @staticmethod
    def _sections(text):
        """[(heading, body)] split on h2s; the lead-in keeps an empty heading."""
        out, heading, buffer = [], '', []
        for line in text.split('\n'):
            if line.startswith('## '):
                out.append((heading, '\n'.join(buffer).strip()))
                heading, buffer = line[3:].strip(), []
            else:
                buffer.append(line)
        out.append((heading, '\n'.join(buffer).strip()))
        return [(head, body) for head, body in out if body]

    def pageFor(self, mapName):
        """The guide page covering a locationTracking map id, if there is one."""
        relative = self.byMapId.get(str(mapName))
        if not relative:
            return None
        for page in self.pages:
            if page.get('file') == relative:
                return page
        return {'title': str(mapName), 'file': relative}

    # ---- what the player is standing in ------------------------------------

    def stepsFor(self, mapName, focus='', budget=1200):
        """The slice of this map's walkthrough that best fits what we're doing.

        `focus` is the current objective. Pages run to a dozen sections and the
        player is only ever in one of them, so the objective picks the section -
        without it the honest choice is the first, which is where you start.
        """
        page = self.pageFor(mapName)
        if page is None:
            return ''
        sections = self._sections(self._body(page['file']))
        if not sections:
            return ''

        # The steps live under "<Place> Walkthrough - ..." headings; the rest of
        # a page is encounter tables, which the player reads from the ROM.
        steps = [s for s in sections if 'walkthrough' in s[0].lower()] or sections
        wanted = _words(focus)
        if wanted:
            heading, body = max(
                steps, key=lambda s: len(wanted & _words(s[0] + ' ' + s[1])))
        else:
            heading, body = steps[0]

        title = page.get('title') or ''
        head = f'{title} — {heading}' if heading else title
        return f'{head}\n{_trim(_readable(body), budget)}'.strip()

    # ---- asking it something -----------------------------------------------

    def lookup(self, term, budget=700, limit=6):
        """Answer `term` from the reference pages, or name it as a place.

        A query matches when every word of it appears, so "bulbasaur nature"
        finds the section about exactly that instead of nothing.
        """
        words = [word for word in re.findall(r"[a-z0-9'.]+", (term or '').lower())
                 if word]
        if not words or not self.available:
            return ''

        def matches(text):
            low = (text or '').lower()
            return all(word in low for word in words)

        # A place first: "mt. moon" is a question about somewhere to go, and
        # the dex would otherwise answer it with everything living there.
        hits = [f'{page["title"]} ({page["section"]}): '
                f'{_trim(page.get("summary") or "", 200)}'
                for page in self.pages if matches(page.get('title'))]

        for entry in self.reference:
            body = self._body(entry['file'])
            # Prose advice is organised by heading - the starter and nature
            # pages answer in paragraphs, not in table rows.
            for heading, chunk in self._sections(body):
                if heading and matches(heading):
                    hits.append(f'{entry["title"]} — {heading}: '
                                f'{_trim(_readable(chunk), 260)}')

            caption = ''
            for line in body.split('\n'):
                stripped = line.strip()
                # Move lists are bullets under a bold type heading, and a bare
                # "Pin Missile" is no answer - carry the heading down to it.
                if stripped.startswith('**') and stripped.endswith('**'):
                    caption = stripped.strip('*')
                    continue
                if stripped.startswith('| ---') or not (
                        stripped.startswith('|') or stripped.startswith('- ')):
                    continue
                if not matches(stripped):
                    continue
                fact = _plain(stripped)
                if caption and stripped.startswith('- '):
                    fact = f'{caption}: {fact}'
                hits.append(f'{entry["title"]}: {fact}')

        return _trim('\n'.join(hits[:limit]), budget)

    def referenceNames(self):
        return [entry['title'] for entry in self.reference]
