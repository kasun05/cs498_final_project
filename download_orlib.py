from pathlib import Path
from urllib.request import urlretrieve


BENCHMARK_DIR = Path("benchmarks/orlib_cap")

# Start small. These are common OR-Library cap-style benchmark names.
INSTANCE_NAMES = [
    "cap41",
    "cap42",
    "cap43",
    "cap44",
    "cap61",
    "cap62",
    "cap63",
    "cap64",
]

BASE_URLS = [
    "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/{name}.txt",
    "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/{name}",
]


def download_instance(name: str) -> None:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    output_path = BENCHMARK_DIR / f"{name}.txt"

    if output_path.exists():
        print(f"Skipping existing file: {output_path}")
        return

    last_error = None

    for template in BASE_URLS:
        url = template.format(name=name)

        try:
            print(f"Trying {url}")
            urlretrieve(url, output_path)

            if output_path.stat().st_size == 0:
                output_path.unlink()
                raise RuntimeError("Downloaded empty file")

            print(f"Saved {output_path}")
            return

        except Exception as exc:
            last_error = exc
            if output_path.exists():
                output_path.unlink()

    raise RuntimeError(f"Failed to download {name}. Last error: {last_error}")


def main() -> None:
    for name in INSTANCE_NAMES:
        download_instance(name)

    print(f"\nDownloaded OR-Library instances to {BENCHMARK_DIR}")


if __name__ == "__main__":
    main()