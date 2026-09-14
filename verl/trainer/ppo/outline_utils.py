class NestedOutlineFormatter:
    """Parses one planner generation ("1. ... 2. ... N. ...") into N outline strings.

    Parsing fails (returns None from `parse_outlines_batched`) unless every marker
    "\\n{i}. " for i in 1..N appears exactly once.
    """

    def __init__(self, num_outlines=4):
        self.num_outlines = num_outlines

    def parse_outlines_batched(self, texts):
        ret = []
        for text in texts:
            try:
                ret.append(self.parse_outlines(text))
            except Exception:
                ret.append(None)
        return ret

    def parse_outlines(self, text):
        if not text.startswith('1. '):
            text = '1. ' + text.split('\n1. ')[-1].strip()
        text = '\n' + text
        for i in range(self.num_outlines):
            assert text.count('\n{:d}. '.format(i + 1)) == 1
        outlines = [text.split(f'\n{i + 1}. ')[-1].split(f'\n{i + 2}. ')[0].strip() for i in range(self.num_outlines)]
        return outlines


def get_outline_formatter(outline_config):
    return NestedOutlineFormatter(outline_config.num_outlines)
