import re
from dataclasses import dataclass, field
from typing import ClassVar


def _find_third_colon(line: str) -> int:
    first = line.find(":")
    second = line.find(":", first + 1)

    return line.find(":", second + 1)


@dataclass
class Experiment:
    donor: str = ""
    segment: str = ""
    classifier: str = ""
    accuracy: str = ""
    avg_time_per_image: str = ""
    macro_f1: str = ""
    macro_f1_per_class: list[str] = field(default_factory=list)

    REGEX: ClassVar[list[re.Pattern[str]]] = [
        re.compile(r"Donor: (?P<data>\w*)"),
        re.compile(r"Segment: (?P<data>\d*:\d*)"),
        re.compile(r"Classifier: (?P<data>\[.*?\])"),
        re.compile(r"Accuracy: (?P<data>\d*\.\d*)"),
        re.compile(r"Macro f1 per class: (?P<data>\[.*\])"),
        re.compile(r"Macro f1: (?P<data>\d*.\d*)"),
        re.compile(r"Average time per image: (?P<data>\d*.\d*)"),
        # re.compile(r"Classification speed: (?P<data>\d*.\d*)"),
    ]

    def update(self, line: str) -> None:
        third_colon_pos = _find_third_colon(line)
        message = line[third_colon_pos + 1 :].strip()

        pattern_index = len(Experiment.REGEX) + 1
        for i, pattern in enumerate(Experiment.REGEX):
            if (result := pattern.match(message)) is not None:
                data = result.group("data")
                pattern_index = i
                break
        else:
            return

        match pattern_index:
            case 0:
                self.donor = data
            case 1:
                self.segment = data
            case 2:
                self.classifier = data
            case 3:
                self.accuracy = data
            case 4:
                self.macro_f1_per_class = data[1:-1].split(", ")
            case 5:
                self.macro_f1 = data
            case 6:
                self.avg_time_per_image = data
