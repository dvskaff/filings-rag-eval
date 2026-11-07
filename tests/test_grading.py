import pytest

from filings.grading import answer_kind, grade, numbers, polarity


@pytest.mark.parametrize(
    ("answer", "gold", "expected"),
    [
        ("Capital expenditure was $1,577 million [1].", "$1577.00", "Correct"),
        ("About $1.58 billion.", "$1577.00", "Correct"),  # other unit, rounded
        ("$1,373 million", "$1577.00", "Incorrect"),
        ("The quick ratio is 0.96.", "0.96", "Correct"),
        ("Operating margin was 36.8%", "36.8%", "Correct"),
        ("I don't know based on the provided filing.", "$1577.00", "Refusal"),
        ("The filing does not include the cash flow statement.", "$1577.00", "Refusal"),
        ("No. Gross margin declined from 19.4% to 18.5%.", "No. Gross margins declined by 0.8%.", "Correct"),
        (
            "Amcor does not have an improving gross margin.",
            "No, it declined",
            "Correct",
        ),  # polarity without "No"
        ("Yes, it improved.", "No, it declined", "Incorrect"),
        ("Form 10-K for 2023 [1]", "$9,870", "Incorrect"),  # citation markers and form names are not numbers
    ],
)
def test_grade(answer, gold, expected):
    assert grade(answer, gold) == expected


def test_helpers():
    assert answer_kind("$1577.00") == "numeric" and answer_kind("Yes. It did.") == "yes/no"
    assert answer_kind("The Consumer Health business segment will be treated as discontinued.") == "text"
    assert numbers("In FY2023 revenue was 1,234.5 million (up 3%) [2]") == [(1234.5, "million"), (3.0, "%")]
    assert polarity("Unfortunately the margin declined.") == "no" and polarity("It improved.") == "yes"
