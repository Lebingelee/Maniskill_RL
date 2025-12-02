import h5py
import argparse
import fnmatch

def inspect_h5_keywords(file_path, keywords):
    """
    打印 .h5 文件中 key 路径包含任一关键词的数据集的 shape 和 dtype。
    """
    keywords = [kw.lower() for kw in keywords]  # 统一转为小写，便于匹配

    def print_if_match(name, obj):
        if isinstance(obj, h5py.Dataset):
            name_lower = name.lower()
            if any(kw in name_lower for kw in keywords):
                print(f"Key: {name:<50} | Shape: {str(obj.shape):<25} | Dtype: {obj.dtype}")

    with h5py.File(file_path, 'r') as f:
        print(f"Inspecting HDF5 file: {file_path}")
        print(f"Keywords: {keywords}")
        print("=" * 90)
        print(f"{'Key':<50} | {'Shape':<25} | Dtype")
        print("-" * 90)
        f.visititems(print_if_match)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect shapes of datasets in an .h5 file whose keys contain given keywords."
    )
    parser.add_argument("file_path", type=str, help="Path to the .h5 file")
    parser.add_argument(
        "-k", "--keywords",
        nargs='+',
        default=['action', 'sensor'],
        help="One or more keywords to filter keys (e.g., -k action sensor_data)"
    )
    args = parser.parse_args()

    inspect_h5_keywords(args.file_path, args.keywords)