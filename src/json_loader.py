import json
from pathlib import Path

import torch.nn

from src.segment import Segment
from src.segment_loader import BaseLoader


class JSON_loader(BaseLoader):
    def __init__(self, file: Path):
        self._file = file

    def load(self) -> Segment:
        with open(self._file, "r", encoding="utf-8") as config_file:
            modules_data = json.load(config_file)

        modules: list[torch.nn.Module] = []
        for module_data in modules_data:
            match module_data["type"]:
                case "Conv":
                    module = torch.nn.Conv2d(
                        module_data["in"],
                        module_data["out"],
                        module_data["kernel_size"],
                        module_data["stride"],
                        module_data["padding"],
                    )
                case "ReLU":
                    module = torch.nn.ReLU(module_data["inplace"])
                case "MaxPool":
                    module = torch.nn.MaxPool2d(
                        module_data["kernel_size"],
                        module_data["stride"],
                        module_data["padding"],
                    )
                case "AdaptiveAvgPool":
                    module = torch.nn.AdaptiveAvgPool2d(module_data["output_size"])
                case "Dropout":
                    module = torch.nn.Dropout(
                        module_data["percent"], module_data["inplace"]
                    )
                case "Linear":
                    module = torch.nn.Linear(
                        module_data["in"], module_data["out"], module_data["bias"]
                    )

            modules.append(module)

        return Segment(modules)
