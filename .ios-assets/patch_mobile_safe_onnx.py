from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def patch_zipmap_model(input_path: Path, output_path: Path) -> None:
    model = onnx.load(str(input_path))
    graph = model.graph

    if not any(opset.domain == "" for opset in model.opset_import):
        model.opset_import.append(helper.make_opsetid("", 13))

    zipmap_node = next((node for node in graph.node if node.op_type == "ZipMap"), None)
    if zipmap_node is None:
        raise ValueError(f"No ZipMap node found in {input_path.name}")

    probability_input = zipmap_node.input[0]
    probability_output = zipmap_node.output[0]

    class_count = 2
    for attribute in zipmap_node.attribute:
        if attribute.name == "classlabels_int64s":
            class_count = len(attribute.ints)
            break
        if attribute.name == "classlabels_strings":
            class_count = len(attribute.strings)
            break

    graph.node.remove(zipmap_node)

    if probability_input != probability_output:
        identity_name = f"{input_path.stem}_zipmap_bypass"
        graph.node.append(
            helper.make_node(
                "Identity",
                inputs=[probability_input],
                outputs=[probability_output],
                name=identity_name,
            )
        )

    for index, output in enumerate(graph.output):
        if output.name == probability_output:
            graph.output.remove(output)
            graph.output.insert(
                index,
                helper.make_tensor_value_info(
                    probability_output,
                    TensorProto.FLOAT,
                    [None, class_count],
                ),
            )
            break
    else:
        graph.output.append(
            helper.make_tensor_value_info(
                probability_output,
                TensorProto.FLOAT,
                [None, class_count],
            )
        )

    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch ONNX classifier outputs to bypass ZipMap for mobile runtimes.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    patch_zipmap_model(args.input, args.output)


if __name__ == "__main__":
    main()
