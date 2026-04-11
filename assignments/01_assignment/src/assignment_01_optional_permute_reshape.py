import torch


def show_tensor_info(name: str, t: torch.Tensor) -> None:
    print(
        f"{name:18s} shape={tuple(t.shape)} stride={t.stride()} "
        f"contiguous={t.is_contiguous()} storage_ptr={t.untyped_storage().data_ptr()}"
    )


def main() -> None:
    print("=== Basis ===")
    base = torch.arange(24).reshape(2, 3, 4)
    permuted = base.permute(1, 0, 2)
    reshaped_base = base.reshape(6, 4)

    show_tensor_info("base", base)
    show_tensor_info("permuted", permuted)
    show_tensor_info("reshaped_base", reshaped_base)

    print("base und permuted teilen Speicher:", base.untyped_storage().data_ptr() == permuted.untyped_storage().data_ptr())
    print("base und reshaped_base teilen Speicher:", base.untyped_storage().data_ptr() == reshaped_base.untyped_storage().data_ptr())

    print("\n=== view vs reshape ===")
    base_view = base.view(6, 4)
    show_tensor_info("base_view", base_view)

    try:
        _ = permuted.view(3, 8)
        print("view auf permuted: funktioniert")
    except RuntimeError as err:
        print("view auf permuted: funktioniert nicht")
        print("Fehler (kurz):", str(err).split("\\n")[0])

    reshaped_permuted = permuted.reshape(3, 8)
    show_tensor_info("reshaped_permuted", reshaped_permuted)
    print(
        "permuted und reshaped_permuted teilen Speicher:",
        permuted.untyped_storage().data_ptr() == reshaped_permuted.untyped_storage().data_ptr(),
    )

if __name__ == "__main__":
    main()