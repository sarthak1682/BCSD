from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence


BINARYCORP_BIN3M_PATTERN = re.compile(
    r"^(?P<binary>.+)-(?P<opt>O[0-3s])-(?P<digest>[0-9a-f]{32})$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class VariantBinary:
    split: str
    project: Optional[str]
    binary_family: str
    variant_label: str
    opt_level: str
    binary_path: Path
    relative_path: str

    @property
    def binary_key(self) -> str:
        if self.project:
            return f"{self.project}/{self.binary_family}"
        return self.binary_family


class BaseCorpusAdapter:
    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = Path(dataset_root)

    def iter_samples(self, split: str) -> Iterator[VariantBinary]:
        raise NotImplementedError


class BinaryCorpBin3mAdapter(BaseCorpusAdapter):
    def iter_samples(self, split: str) -> Iterator[VariantBinary]:
        split_dir = self.dataset_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

        grouped: Dict[str, List[tuple[Path, str, str]]] = defaultdict(list)
        for path in sorted(split_dir.iterdir()):
            if not path.is_file() or path.is_symlink():
                continue
            match = BINARYCORP_BIN3M_PATTERN.match(path.name)
            if not match:
                continue

            binary_family = match.group("binary")
            opt_level = match.group("opt")
            opt_level = "Os" if opt_level.lower() == "os" else opt_level.upper()
            digest = match.group("digest")
            grouped[binary_family].append((path, opt_level, digest))

        for binary_family in sorted(grouped):
            entries = grouped[binary_family]
            opt_counts = Counter(opt_level for _path, opt_level, _digest in entries)
            for path, opt_level, digest in entries:
                if opt_counts[opt_level] == 1:
                    variant_label = opt_level
                else:
                    variant_label = f"{opt_level}@{digest[:8]}"

                yield VariantBinary(
                    split=split,
                    project=None,
                    binary_family=binary_family,
                    variant_label=variant_label,
                    opt_level=opt_level,
                    binary_path=path,
                    relative_path=str(path.relative_to(self.dataset_root)),
                )


class BenchsetAdapter(BaseCorpusAdapter):
    DEFAULT_ARCHITECTURES: Sequence[str] = ("x86_64", "arm64", "mips64", "powerpc64")

    def __init__(
        self,
        dataset_root: Path,
        architectures: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(dataset_root)
        self.architectures = tuple(
            sorted(architectures or self.DEFAULT_ARCHITECTURES, key=len, reverse=True)
        )

    def _consume_arch(self, tokens: Sequence[str]) -> tuple[str, int]:
        joined = "_".join(tokens)
        for arch in self.architectures:
            if joined == arch or joined.startswith(f"{arch}_"):
                return arch, len(arch.split("_"))
        raise ValueError(
            f"Unable to infer architecture from tokens {tokens!r}. "
            f"Known architectures: {', '.join(self.architectures)}"
        )

    def _parse_binary_name(self, project_name: str, binary_name: str) -> tuple[str, str, str]:
        prefix = f"{project_name}_"
        if not binary_name.startswith(prefix):
            raise ValueError(
                f"Benchmark binary {binary_name!r} must start with project prefix {prefix!r}"
            )

        remainder = binary_name[len(prefix) :]
        parts = remainder.split("_")
        if len(parts) < 5:
            raise ValueError(
                "Benchmark binary names must follow "
                "<project>_<compiler>_<opt>_<arch>_<obf>_<binaryname>"
            )

        compiler = parts[0]
        opt_level = parts[1]
        arch, arch_len = self._consume_arch(parts[2:])
        tail = parts[2 + arch_len :]
        if len(tail) < 2:
            raise ValueError(
                f"Unable to parse obfuscation and binary name from benchmark binary {binary_name!r}"
            )

        obf = tail[0]
        binary_family = "_".join(tail[1:])
        variant_label = "_".join((compiler, opt_level, arch, obf))
        return binary_family, variant_label, opt_level

    def iter_samples(self, split: str) -> Iterator[VariantBinary]:
        split_dir = self.dataset_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

        for project_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            for path in sorted(child for child in project_dir.iterdir() if child.is_file() and not child.is_symlink()):
                binary_family, variant_label, opt_level = self._parse_binary_name(
                    project_name=project_dir.name,
                    binary_name=path.name,
                )
                yield VariantBinary(
                    split=split,
                    project=project_dir.name,
                    binary_family=binary_family,
                    variant_label=variant_label,
                    opt_level=opt_level,
                    binary_path=path,
                    relative_path=str(path.relative_to(self.dataset_root)),
                )


def create_adapter(dataset_type: str, dataset_root: Path) -> BaseCorpusAdapter:
    if dataset_type == "binarycorp-bin3m":
        return BinaryCorpBin3mAdapter(dataset_root)
    if dataset_type == "benchset":
        return BenchsetAdapter(dataset_root)
    raise ValueError(f"Unsupported dataset type: {dataset_type}")
