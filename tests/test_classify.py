from docchunker.classify import classify

CODE = """def add(a, b):
    total = a + b
    return total

class Foo:
    pass"""

JS = """function go(x) {
  const y = x * 2;
  return y;
}"""

PROSE = """Our dog detection model reaches ninety percent accuracy on the validation set.
The app then lets the owner book a consultation with a nearby veterinarian, which saves time."""

IF_PROSE = """If the dog keeps scratching, for example after a walk, a vet visit is recommended.
While waiting, keep the area clean and dry, and avoid using any human medication."""


def test_python_is_code():
    assert classify(CODE) == "code"


def test_js_is_code():
    assert classify(JS) == "code"


def test_fenced_is_code():
    assert classify("Example:\n```\nls -la\n```") == "code"


def test_prose_is_text():
    assert classify(PROSE) == "text"


def test_prose_starting_with_keywords_is_text():
    # v1 flagged any line starting with if/while/for as code
    assert classify(IF_PROSE) == "text"


def test_single_line_and_empty():
    assert classify("") == "text"
    assert classify("x = 1") == "text"
