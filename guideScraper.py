"""
Mirror IGN's Pokemon FireRed/LeafGreen walkthrough as local Markdown.

The wiki is a Next.js app, but the whole article body is already sitting in the
page's __NEXT_DATA__ blob as a list of HTML fragments, so nothing here needs a
browser or a language model: fetch, pull the JSON out, convert the HTML, write
the file.

    python guideScraper.py                  # scrape everything into guides/
    python guideScraper.py --list           # print the page order, one request
    python guideScraper.py --only Route_1   # redo a single page
    python guideScraper.py --refresh        # ignore the cached HTML, refetch

Output layout (guides/ is gitignored - it is a rebuildable mirror, not source):

    guides/README.md                                  what this is
    guides/index.md                                   every page, in order
    guides/index.json                                 the same, for code
    guides/walkthrough/01-boulder-badge/03-viridian-city.md
    guides/.cache/Viridian_City.html                  raw HTML, so reruns are free

Pages come from the walkthrough index, in the order it lists them, so the
numbering matches the intended play order. Output is text only - the steps -
with IGN's screenshots and checklist widgets dropped rather than linked. Every
file carries YAML frontmatter naming the locationTracking maps it covers
(`3-1-ViridianCity` -> bank 3, map 1), and index.json has a byMapId lookup, so
the player can pull up the guide for whatever map it is standing on.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

WIKI = 'https://www.ign.com/wikis/pokemon-firered-leafgreen-version'
START_PAGE = 'Walkthrough'
USER_AGENT = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
              'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36')

HERE = os.path.dirname(os.path.abspath(__file__))
MAP_IDS_PATH = os.path.join(HERE, 'locationTracking', 'connectionData', 'mapIds.json')

MAX_PAGES = 250

# Locations the walkthrough covers but leaves out of its own index, as
# (page it follows, label, page name).
EXTRA_STEPS = [('Green_Path', 'Outcast Island', 'Outcast_Island')]

# Lookup tables that are not part of the walk, but that the walk keeps pointing
# at - the dex with its catch locations, what the other version gets, what
# every TM and HM does, and the two starter questions the first turns hinge on.
REFERENCE_PAGES = [
    'Pokedex_-_All_Pokemon_List_and_Locations',
    'Version_Exclusive_Pokemon',
    'Techniques,_HMs_and_TMs_List',
    'Best_Starter_to_Choose',
    'Best_Natures_for_Bulbasaur,_Charmander_and_Squirtle',
]


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------

def sslContext():
    """python.org builds on macOS ship no root certificates - borrow certifi's."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetchPage(name, cacheDir, refresh=False, delay=1.0, quiet=False):
    """Return the raw HTML for a wiki page, going to the network at most once."""
    os.makedirs(cacheDir, exist_ok=True)
    cached = os.path.join(cacheDir, name.replace('/', '_') + '.html')
    if os.path.exists(cached) and not refresh:
        with open(cached, 'r', encoding='utf-8') as f:
            return f.read()

    url = f'{WIKI}/{urllib.parse.quote(name, safe="_-.,()!%")}'
    if not quiet:
        print(f'  fetching {url}', file=sys.stderr)
    request = urllib.request.Request(url, headers={
        'User-Agent': USER_AGENT,
        'Accept': 'text/html,application/xhtml+xml',
        'Accept-Language': 'en-US,en;q=0.9',
    })
    with urllib.request.urlopen(request, timeout=45, context=sslContext()) as response:
        html = response.read().decode('utf-8', 'replace')
    with open(cached, 'w', encoding='utf-8') as f:
        f.write(html)
    if delay:
        time.sleep(delay)
    return html


def nextData(html):
    """The __NEXT_DATA__ payload, which holds the article body and its metadata."""
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html, re.S)
    if not match:
        raise ValueError('no __NEXT_DATA__ on the page - IGN changed its markup')
    return json.loads(match.group(1))


def pageRecord(html):
    """Flatten a fetched page into the handful of fields the mirror needs."""
    props = nextData(html)['props']['pageProps']['page']
    page = props['page']
    body = ''.join(block['values'].get('html', '')
                   for block in page.get('htmlEntities') or []
                   if block.get('name') == 'html')
    return {
        'name': (props.get('chapter') or page.get('title') or '').replace(' ', '_'),
        'title': page.get('title') or '',
        'summary': unescapeEntities(props.get('description') or ''),
        'updated': page.get('updatedAt') or '',
        'ignNext': urllib.parse.unquote(
            ((page.get('nextPage') or {}).get('url') or '').strip('/')),
        'ignPrev': urllib.parse.unquote(
            ((page.get('prevPage') or {}).get('url') or '').strip('/')),
        'next': '',
        'prev': '',
        'html': body,
    }


def unescapeEntities(text):
    import html as htmlmod
    return htmlmod.unescape(text)


# --------------------------------------------------------------------------
# HTML -> Markdown
# --------------------------------------------------------------------------

class DomBuilder(HTMLParser):
    """Just enough of a DOM to render tables and nested lists correctly."""

    VOID = {'br', 'img', 'hr', 'input', 'meta', 'link', 'source', 'col'}
    # Dropped outright: `checkbox` is IGN's "have you caught it yet?" widget,
    # and `img` is a screenshot whose alt text is only ever its own filename.
    # The mirror is text - the steps - so neither earns a place in it.
    DROP = {'script', 'style', 'noscript', 'checkbox', 'img'}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = {'tag': 'root', 'attrs': {}, 'children': []}
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = {'tag': tag, 'attrs': dict(attrs), 'children': []}
        self.stack[-1]['children'].append(node)
        if tag not in self.VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1]['children'].append(
            {'tag': tag, 'attrs': dict(attrs), 'children': []})

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i]['tag'] == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        self.stack[-1]['children'].append(data)


def parseHtml(html):
    builder = DomBuilder()
    builder.feed(html)
    builder.close()
    return builder.root


def inlineText(node, ctx):
    """Render a node as a single line of Markdown."""
    if isinstance(node, str):
        return re.sub(r'\s+', ' ', node)

    tag = node['tag']
    if tag in DomBuilder.DROP:
        return ''
    if tag == 'br':
        return '\n'

    inner = ''.join(inlineText(child, ctx) for child in node['children'])
    stripped = inner.strip()
    marks = {'b': '**', 'strong': '**', 'i': '*', 'em': '*', 'code': '`'}
    if tag in marks:
        if not stripped:
            return ''
        # Markdown emphasis cannot hold whitespace against its markers, so move
        # it outside instead of dropping it - otherwise the text after a closing
        # </i> runs straight into it.
        lead = inner[:len(inner) - len(inner.lstrip())]
        trail = inner[len(inner.rstrip()):]
        return f'{lead}{marks[tag]}{stripped}{marks[tag]}{trail}'
    if tag == 'a':
        target = ctx['link'](node['attrs'].get('href', ''))
        text = stripped or target or ''
        if not text:
            return ''
        return f'[{text}]({target})' if target else text
    return inner


def renderTable(node, ctx):
    """A wiki table becomes a Markdown table, with any colspan banner as a caption."""
    rows = []

    def collect(parent):
        for child in parent['children']:
            if not isinstance(child, dict):
                continue
            if child['tag'] == 'tr':
                cells = []
                for cell in child['children']:
                    if not isinstance(cell, dict) or cell['tag'] not in ('td', 'th'):
                        continue
                    # A cell's <br>s are meaningful - they separate the items
                    # or the level range - but a table row cannot hold them.
                    text = inlineText(cell, ctx).replace('|', '\\|')
                    text = re.sub(r'[ \t]*\n[ \t]*', '; ', text)
                    text = re.sub(r'(?:; )+', '; ', text).strip(' ;')
                    text = re.sub(r'\s+', ' ', text)
                    span = cell['attrs'].get('colspan') or '1'
                    cells.append((cell['tag'], text, int(span) if span.isdigit() else 1))
                if cells:
                    rows.append(cells)
            elif child['tag'] in ('thead', 'tbody', 'tfoot'):
                collect(child)

    collect(node)
    if not rows:
        return []

    rows = [row for row in rows if any(text for _, text, _ in row)]

    blocks = []
    while rows and len(rows[0]) == 1 and rows[0][0][0] == 'th':
        caption = rows.pop(0)[0][1]
        if not caption:
            continue
        # The caption is often already bold in the source; do not bold it twice.
        bold = re.fullmatch(r'\*+(.+?)\*+', caption)
        if bold and '**' not in bold.group(1):
            caption = bold.group(1)
        blocks.append(f'**{caption}**')
    if not rows:
        return blocks

    # A headerless single row is a layout, not data, and comes in two shapes.
    if len(rows) == 1 and not any(tag == 'th' for tag, _, _ in rows[0]):
        cells = [text for _, text, _ in rows[0] if text]
        # A name beside its description: the item pop-ups. Keep the pair on one
        # line - splitting them loses which of the two describes the other.
        if len(cells) == 2:
            blocks.append(f'**{cells[0]}** — {cells[1]}')
            return blocks
        # More cells, each holding several entries: the move lists, set out in
        # four newspaper columns. Read them back as one list.
        if any('; ' in text for text in cells):
            items = [part.strip() for text in cells
                     for part in text.split('; ') if part.strip()]
            if items:
                blocks.append('\n'.join(f'- {item}' for item in items))
            return blocks

    width = max(sum(span for _, _, span in row) for row in rows)

    def expand(row):
        # A lone full-width cell is a banner splitting the table into sections
        # ("Old Rod", "Pallet Town Items") - bold it so it does not read as data.
        if len(row) == 1 and row[0][2] >= width:
            tag, text, _ = row[0]
            if tag == 'th' and text:
                text = f'**{text}**'
            return [text] + [''] * (width - 1)
        out = []
        for _, text, span in row:
            out.append(text)
            out.extend([''] * (span - 1))
        return (out + [''] * width)[:width]

    header = rows[0]
    if all(tag == 'th' for tag, _, _ in header):
        body = rows[1:]
        headerCells = expand(header)
    else:
        body = rows
        headerCells = [''] * width

    bodyCells = [expand(row) for row in body]

    # Columns that are empty all the way down are IGN's checklist widgets and
    # spacers - drop them rather than ship a column of blanks.
    keep = [index for index in range(width)
            if any(row[index] for row in bodyCells)] or list(range(width))
    headerCells = [headerCells[index] for index in keep]
    bodyCells = [[row[index] for index in keep] for row in bodyCells]

    # What is left may be a single column - a list of items wearing a table
    # costume. Say it as a list, and split the cells the <br>s had joined.
    if len(keep) == 1:
        if headerCells[0]:
            blocks.append(f'**{headerCells[0]}**')
        items = [part.strip() for row in bodyCells
                 for part in row[0].split('; ') if part.strip()]
        if items:
            blocks.append('\n'.join(f'- {item}' for item in items))
        return blocks

    lines = ['| ' + ' | '.join(headerCells) + ' |',
             '| ' + ' | '.join(['---'] * len(keep)) + ' |']
    lines += ['| ' + ' | '.join(row) + ' |' for row in bodyCells]
    blocks.append('\n'.join(lines))
    return blocks


def renderList(node, ctx, depth):
    ordered = node['tag'] == 'ol'
    lines = []
    number = 1
    for item in node['children']:
        if not isinstance(item, dict) or item['tag'] != 'li':
            continue
        blocks = renderBlocks(item['children'], ctx, depth + 1)
        if not blocks:
            continue
        marker = f'{number}.' if ordered else '-'
        number += 1
        first, *rest = '\n\n'.join(blocks).split('\n')
        pad = ' ' * (len(marker) + 1)
        lines.append(f'{marker} {first}')
        lines.extend(pad + line if line else '' for line in rest)
    return '\n'.join(lines)


def renderBlocks(nodes, ctx, depth=0):
    """Walk a node list, emitting one string per Markdown block."""
    out = []
    buffer = []

    def flush():
        text = ''.join(buffer).strip()
        del buffer[:]
        if text:
            out.append(re.sub(r'\n{2,}', '\n', text))

    for node in nodes:
        if isinstance(node, str):
            buffer.append(re.sub(r'\s+', ' ', node))
            continue
        tag = node['tag']
        if tag in DomBuilder.DROP:
            continue
        if tag == 'table':
            flush()
            out.extend(renderTable(node, ctx))
        elif tag in ('ul', 'ol'):
            flush()
            rendered = renderList(node, ctx, depth)
            if rendered:
                out.append(rendered)
        elif re.fullmatch(r'h[1-6]', tag):
            flush()
            text = inlineText(node, ctx).strip()
            if text:
                out.append('#' * int(tag[1]) + ' ' + text)
        elif tag in ('p', 'div', 'section', 'blockquote', 'center'):
            flush()
            inner = renderBlocks(node['children'], ctx, depth)
            if 'box' in node['attrs'].get('class', '') or tag == 'blockquote':
                inner = ['\n'.join('> ' + line if line else '>'
                                   for line in block.split('\n')) for block in inner]
            out.extend(inner)
        elif tag == 'hr':
            flush()
            out.append('---')
        else:
            buffer.append(inlineText(node, ctx))
    flush()
    return [block for block in out if block.strip()]


def htmlToMarkdown(html, linkResolver):
    return '\n\n'.join(renderBlocks(parseHtml(html)['children'], {'link': linkResolver}))


def plainText(node):
    """A node's text with links and emphasis dropped - for labels and headings."""
    return re.sub(r'\*+', '', inlineText(node, {'link': lambda href: None})).strip()


# --------------------------------------------------------------------------
# matching pages to locationTracking maps
# --------------------------------------------------------------------------

# Where IGN and the repo disagree on what a place is called.
MAP_ALIASES = {
    'silph company': 'silph co',
    "team rocket's hideout": 'rocket hideout',
}


def splitName(name):
    """`1-2-MtMoon_B1F` -> ['mt', 'moon', 'b1f']; 'Mt. Moon' -> ['mt', 'moon']."""
    name = re.sub(r'^\d+-\d+-', '', name)
    name = name.replace('_', ' ')
    name = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name)
    name = re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', name)
    return [word for word in re.split(r'[^A-Za-z0-9]+', name.lower()) if word]


def loadMapIndex(path=MAP_IDS_PATH):
    """Map every known map id to the word list its name reduces to."""
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        mapIds = json.load(f)
    return [(key, banks, splitName(key)) for key, banks in sorted(mapIds.items())]


def matchMaps(title, mapIndex):
    """Every map id whose name is the page title, or the title plus a floor/wing."""
    words = splitName(MAP_ALIASES.get(title.strip().lower(), title))
    if not words:
        return []
    squashed = ''.join(words)
    endsInDigit = squashed[-1].isdigit()

    matches = []
    for key, banks, keyWords in mapIndex:
        keySquashed = ''.join(keyWords)
        if keySquashed == squashed:
            hit = True
        elif keyWords[:len(words)] == words:
            hit = True
        elif keySquashed.startswith(squashed):
            # Route 1 must not swallow Route 10, but S.S. Anne must still reach
            # SSAnne_1F_Corridor - only guard the numbered names.
            hit = not endsInDigit
        else:
            hit = False
        if hit:
            matches.append({'id': key, 'banks': [list(pair) for pair in banks]})
    return matches


# --------------------------------------------------------------------------
# crawling
# --------------------------------------------------------------------------

def slugify(text):
    slug = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
    return slug or 'page'


def parseIndex(html):
    """The walkthrough index, as [(section title, [(label, page name)])]."""
    body = pageRecord(html)['html']
    sections = []
    current = None

    def scan(node):
        nonlocal current
        for child in node['children']:
            if not isinstance(child, dict):
                continue
            tag = child['tag']
            if re.fullmatch(r'h[1-6]', tag):
                title = plainText(child)
                if title:
                    current = (title, [])
                    sections.append(current)
            elif tag == 'table':
                links = []
                stack = [child]
                while stack:
                    node2 = stack.pop(0)
                    for kid in node2['children']:
                        if not isinstance(kid, dict):
                            continue
                        if kid['tag'] == 'a':
                            page = pageNameFromHref(kid['attrs'].get('href', ''))
                            label = plainText(kid)
                            if page:
                                links.append((label or page.replace('_', ' '), page))
                        stack.append(kid)
                if links and current:
                    current[1].extend(links)
            else:
                scan(child)

    scan(parseHtml(body))
    return [(title, pages) for title, pages in sections if pages]


def pageNameFromHref(href):
    """`/wikis/pokemon-leafgreen-version/Diglett%27s_Cave#items` -> `Diglett's_Cave`."""
    match = re.match(r'^(?:https?://www\.ign\.com)?/wikis/[^/]+/([^#?]+)', href or '')
    return urllib.parse.unquote(match.group(1)) if match else None


def crawl(outDir, delay=1.0, refresh=False, limit=None, quiet=False):
    """Fetch every page the walkthrough index lists, in the order it lists them.

    The wiki's own next-page pointers look like the natural spine but they are
    not trustworthy - Viridian City points back at Pallet Town - so the index
    tables are the running order, and the chain is only reported on.

    Returns (outline, pages): the outline is the walkthrough in order and
    includes revisits, while pages holds one record per distinct wiki page.
    """
    cacheDir = os.path.join(outDir, '.cache')
    sections = parseIndex(fetchPage(START_PAGE, cacheDir, refresh, delay, quiet))

    # Each badge heading is itself a page, sitting in front of its locations.
    steps = []
    for index, (title, entries) in enumerate(sections, start=1):
        steps.append((index, title, title, title.replace(' ', '_')))
        for label, page in entries:
            steps.append((index, title, label, page))

    for after, label, name in EXTRA_STEPS:
        for position, step in enumerate(steps):
            if step[3] == after:
                steps.insert(position + 1, (step[0], step[1], label, name))
                break

    listed = {page for _, _, _, page in steps}
    pages = {}
    outline = []
    counters = {}

    for index, section, label, name in steps:
        if name in pages:
            pages[name]['revisits'].append({'section': section, 'label': label})
            outline.append({'page': name, 'label': label, 'section': section,
                            'revisit': True})
            continue
        if limit and len(pages) >= limit:
            break
        try:
            record = pageRecord(fetchPage(name, cacheDir, refresh, delay, quiet))
        except urllib.error.HTTPError as error:
            print(f'  skipping {name}: IGN returned {error.code}', file=sys.stderr)
            continue

        counters[section] = counters.get(section, 0) + 1
        record['name'] = name
        record['title'] = record['title'] or label
        record['section'] = section
        record['order'] = counters[section]
        record['revisits'] = []
        record['file'] = os.path.join(
            'walkthrough', f'{index:02d}-{slugify(section)}',
            f'{counters[section]:02d}-{slugify(record["title"])}.md')
        pages[name] = record
        outline.append({'page': name, 'label': label, 'section': section,
                        'revisit': False})

    # Walkthrough order is what the player actually wants to follow next, so it
    # replaces IGN's own prev/next on the page.
    for position, entry in enumerate(outline):
        if entry['revisit']:
            continue
        record = pages[entry['page']]
        record['prev'] = outline[position - 1]['page'] if position else ''
        record['next'] = (outline[position + 1]['page']
                          if position + 1 < len(outline) else '')

    strays = sorted({record['ignNext'] for record in pages.values()
                     if record['ignNext'] and record['ignNext'] not in listed})
    if strays and not quiet:
        print('  pages linked from the guide but not in its index: '
              + ', '.join(strays), file=sys.stderr)

    return outline, list(pages.values())


def fetchReference(outDir, delay=1.0, refresh=False, quiet=False):
    """The standalone lookup pages, which sit outside the walkthrough order."""
    cacheDir = os.path.join(outDir, '.cache')
    records = []
    for name in REFERENCE_PAGES:
        try:
            record = pageRecord(fetchPage(name, cacheDir, refresh, delay, quiet))
        except urllib.error.HTTPError as error:
            print(f'  skipping {name}: IGN returned {error.code}', file=sys.stderr)
            continue
        record['name'] = name
        record['title'] = record['title'] or name.replace('_', ' ')
        record['section'] = 'Reference'
        record['order'] = len(records) + 1
        record['revisits'] = []
        record['file'] = os.path.join('reference', f'{slugify(record["title"])}.md')
        records.append(record)
    return records


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------

def yamlValue(value):
    if isinstance(value, int):
        return str(value)
    text = str(value).replace('\\', '\\\\').replace('"', '\\"')
    return f'"{text}"'


def frontmatter(fields):
    lines = ['---']
    for key, value in fields.items():
        if value in ('', None, []):
            continue
        if key == 'maps':
            lines.append('maps:')
            for entry in value:
                lines.append(f'  - id: {yamlValue(entry["id"])}')
                banks = ', '.join(f'[{b}, {n}]' for b, n in entry['banks'])
                lines.append(f'    banks: [{banks}]')
        elif key == 'revisits':
            lines.append('revisits:')
            for entry in value:
                lines.append(f'  - section: {yamlValue(entry["section"])}')
                lines.append(f'    label: {yamlValue(entry["label"])}')
        else:
            lines.append(f'{key}: {yamlValue(value)}')
    lines.append('---')
    return '\n'.join(lines)


def writePages(outDir, pages, mapIndex):
    byName = {page['name']: page for page in pages}
    written = []

    for page in pages:
        path = os.path.join(outDir, page['file'])
        os.makedirs(os.path.dirname(path), exist_ok=True)

        def resolve(href, _page=page):
            if not href or href.startswith('#'):
                return None
            target = pageNameFromHref(href)
            if target and target in byName:
                relative = os.path.relpath(
                    os.path.join(outDir, byName[target]['file']),
                    os.path.dirname(os.path.join(outDir, _page['file'])))
                anchor = href.split('#', 1)[1] if '#' in href else ''
                return relative + (f'#{anchor}' if anchor else '')
            if target:
                return f'{WIKI}/{target}'
            if href.startswith('http'):
                return href
            return None

        page['maps'] = matchMaps(page['title'], mapIndex)
        head = frontmatter({
            'title': page['title'],
            'section': page['section'],
            'order': page['order'],
            'page': page['name'],
            'source': f'{WIKI}/{page["name"]}',
            'updated': page['updated'],
            'prev': page.get('prev', ''),
            'next': page.get('next', ''),
            'summary': page['summary'],
            'maps': page['maps'],
            'revisits': page.get('revisits', []),
        })
        body = htmlToMarkdown(page['html'], resolve)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f'{head}\n\n# {page["title"]}\n\n{body}\n')
        written.append(page)

    return written


def writeIndexes(outDir, outline, pages, reference=()):
    byName = {page['name']: page for page in pages}

    lines = ['# Pokemon FireRed / LeafGreen walkthrough',
             '',
             'Every page of the walkthrough, in the order it is meant to be '
             f'played. Mirrored from [IGN]({WIKI}/{START_PAGE}). A step marked '
             '_(revisit)_ points back at a page listed earlier.',
             '']
    if reference:
        lines += ['## Reference', '',
                  'Lookup tables, not part of the walk.', '']
        for page in reference:
            lines.append(f'- [{page["title"]}]({page["file"]}) — {page["summary"]}')
    currentSection = None
    for step in outline:
        page = byName.get(step['page'])
        if not page:
            continue
        if step['section'] != currentSection:
            currentSection = step['section']
            lines += ['', f'## {currentSection}', '']
        maps = ', '.join(entry['id'] for entry in page.get('maps', []))
        notes = []
        if step['revisit']:
            notes.append('revisit')
        if maps:
            notes.append(f'maps: {maps}')
        suffix = f' — {" — ".join(notes)}' if notes else ''
        lines.append(f'- [{step["label"]}]({page["file"]}){suffix}')
    lines.append('')
    with open(os.path.join(outDir, 'index.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    # A city page claims its own interiors, so a map id can have several
    # candidates. Send the lookup to the most specific page that claims it -
    # 6-2-PewterCity_Gym belongs to the Gym page, not to Pewter City.
    candidates = {}
    for page in pages:
        for entry in page.get('maps', []):
            candidates.setdefault(entry['id'], []).append(page)
    byMapId = {}
    for mapId, owners in sorted(candidates.items()):
        best = max(owners, key=lambda page: len(splitName(page['title'])))
        byMapId[mapId] = best['file']

    payload = {
        'source': f'{WIKI}/{START_PAGE}',
        'reference': [{
            'title': page['title'],
            'page': page['name'],
            'file': page['file'],
            'summary': page['summary'],
        } for page in reference],
        'walkthrough': [{
            'label': step['label'],
            'page': step['page'],
            'section': step['section'],
            'revisit': step['revisit'],
            'file': byName[step['page']]['file'],
        } for step in outline if step['page'] in byName],
        'pages': [{
            'title': page['title'],
            'page': page['name'],
            'section': page['section'],
            'order': page['order'],
            'file': page['file'],
            'summary': page['summary'],
            'maps': page.get('maps', []),
        } for page in pages],
        'byMapId': byMapId,
    }
    with open(os.path.join(outDir, 'index.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
        f.write('\n')


README = '''# guides/

A local Markdown mirror of IGN's Pokemon FireRed/LeafGreen walkthrough, for the
player to read while it plays. Generated by `guideScraper.py` in the repo root -
this whole directory is gitignored and rebuildable, so do not hand-edit it:

    python guideScraper.py            # rebuild from the HTML cache
    python guideScraper.py --refresh  # refetch from IGN first

## Layout

- `index.md` - every page in play order, grouped by badge.
- `index.json` - the same list for code, plus `byMapId`, which maps a
  locationTracking map id (`3-1-ViridianCity`) straight to a guide file.
- `walkthrough/<NN-badge>/<NN-place>.md` - one file per location, numbered in
  the order the walkthrough visits them.
- `reference/` - lookup tables that are not part of the walk: the dex with a
  catch location for every Pokemon, the version exclusives, and what every TM
  and HM does. Listed under `reference` in index.json.
- `.cache/` - the raw HTML, so a rebuild costs no requests.

## Reading a page

Each file opens with YAML frontmatter:

    title, section, order   where the page sits in the walkthrough
    page, source, updated   where it came from on IGN
    prev, next              the neighbouring pages, by IGN page name
    summary                 IGN's own one-line description
    maps                    locationTracking map ids this page covers

`maps` is the join to the rest of the repo: it is matched by name against
`locationTracking/connectionData/mapIds.json`, so a page titled "Mt. Moon"
carries `1-1-MtMoon_1F`, `1-2-MtMoon_B1F` and `1-3-MtMoon_B2F`, each with the
`[bank, number]` pairs the emulator reports.

## What is in a page

Text only: the walkthrough steps, the trainer and encounter tables, and the
item lists. IGN's screenshots and checklist widgets are dropped outright, not
linked - a screenshot's alt text is only ever its own filename, so a link to
one is dead weight to anything reading this.

Content is IGN's, reproduced here for local reference only.
'''


# --------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--out', default=os.path.join(HERE, 'guides'),
                        help='output directory (default: guides/)')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='seconds to wait between requests (default: 1)')
    parser.add_argument('--refresh', action='store_true',
                        help='refetch pages instead of using the HTML cache')
    parser.add_argument('--limit', type=int,
                        help='stop after this many pages')
    parser.add_argument('--only', metavar='PAGE',
                        help='convert a single IGN page name, e.g. Route_1')
    parser.add_argument('--list', action='store_true',
                        help='print the walkthrough order and exit')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)

    outDir = args.out
    cacheDir = os.path.join(outDir, '.cache')
    mapIndex = loadMapIndex()

    if args.list:
        sections = parseIndex(fetchPage(START_PAGE, cacheDir, args.refresh,
                                        args.delay, args.quiet))
        for title, entries in sections:
            print(f'\n{title}')
            for label, page in entries:
                print(f'  {label:<28} {page}')
        return 0

    if args.only:
        record = pageRecord(fetchPage(args.only, cacheDir, args.refresh,
                                      args.delay, args.quiet))
        record.update({'name': record['name'] or args.only, 'section': 'Walkthrough',
                       'order': 1, 'sectionDir': 'loose',
                       'file': os.path.join('walkthrough', 'loose',
                                            f'{slugify(record["title"])}.md')})
        writePages(outDir, [record], mapIndex)
        print(f'wrote {os.path.join(outDir, record["file"])}')
        return 0

    os.makedirs(outDir, exist_ok=True)
    outline, pages = crawl(outDir, args.delay, args.refresh, args.limit, args.quiet)
    if not pages:
        print('no pages found - IGN may have changed its markup', file=sys.stderr)
        return 1
    reference = fetchReference(outDir, args.delay, args.refresh, args.quiet)

    # Written together so a walkthrough page linking to the dex resolves to the
    # local copy rather than back out to IGN.
    writePages(outDir, pages + reference, mapIndex)
    writeIndexes(outDir, outline, pages, reference)
    with open(os.path.join(outDir, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(README)

    matched = sum(1 for page in pages if page.get('maps'))
    print(f'wrote {len(pages)} pages to {outDir} '
          f'({matched} matched to locationTracking maps)')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except urllib.error.HTTPError as error:
        print(f'IGN returned {error.code} for {error.url}', file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
