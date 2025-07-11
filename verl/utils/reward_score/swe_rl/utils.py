from enum import Enum
from pathlib import Path
from typing import Optional
import cydifflib
import re

from pydantic import BaseModel, Field

# adapted from https://github.com/SWE-bench/SWE-bench/blob/main/swebench/harness/utils.py
PATCH_PATTERN = re.compile(
    r"(?:diff[\w\_\.\ \/\-]+\n)?\-\-\-\s+a\/(?:.*?)\n\+\+\+\s+b\/(?:.*?)(?=diff\ |\-\-\-\ a\/|\Z)",
    re.DOTALL,
)
PATCH_FILE_PATTERN = re.compile(r"\-\-\-\s+a\/(?:.+)\n\+\+\+\s+b\/(?:.+)")
PATCH_HUNK_PATTERN = re.compile(
    r"\@\@\s+\-(\d+),(\d+)\s+\+(\d+),(\d+)\s+\@\@(.+?)(?=diff\ |\-\-\-\ a\/|\@\@\ \-|\Z)",
    re.DOTALL,
)
COMMENT_LINE_PATTERN = re.compile(r"^[+-][ \t]*#.*$")


def strip_comment_lines(patch: str) -> str:
    lines = patch.splitlines(keepends=True)
    filtered = [
        ln for ln in lines if not COMMENT_LINE_PATTERN.match(ln)
    ]
    return "".join(filtered)

def get_first_idx(charlist):
    """Get index of first occurrence of "-" or "+" in charlist"""
    first_min = charlist.index("-") if "-" in charlist else len(charlist)
    first_plus = charlist.index("+") if "+" in charlist else len(charlist)
    return min(first_min, first_plus)


def get_last_idx(charlist):
    """Get index of last occurrence of "-" or "+" in charlist"""
    char_idx = get_first_idx(charlist[::-1])
    last_idx = len(charlist) - char_idx
    return last_idx + 1


def strip_content(hunk):
    """Remove trailing non +/- lines and trailing whitespace per line per hunk"""
    first_chars = list(map(lambda x: None if not len(x) else x[0], hunk.split("\n")))
    first_idx = get_first_idx(first_chars)
    last_idx = get_last_idx(first_chars)
    new_lines = list(map(lambda x: x.rstrip(), hunk.split("\n")[first_idx:last_idx]))
    # should leave one space for empty context lines
    new_lines = [line if line.strip() else " " for line in new_lines]
    new_hunk = "\n" + "\n".join(new_lines) + "\n"
    return new_hunk, first_idx - 1


def get_hunk_stats(pre_start, pre_len, post_start, post_len, hunk, total_delta):
    """Recalculate hunk start/end position and diff delta"""
    stats = {"context": 0, "added": 0, "subtracted": 0}
    hunk = hunk.split("\n", 1)[-1].strip("\n")
    for line in hunk.split("\n"):
        if line.startswith("-"):
            stats["subtracted"] += 1
        elif line.startswith("+"):
            stats["added"] += 1
        else:
            stats["context"] += 1
    context = stats["context"]
    added = stats["added"]
    subtracted = stats["subtracted"]
    pre_len = context + subtracted
    post_start = pre_start + total_delta
    post_len = context + added
    total_delta = total_delta + (post_len - pre_len)
    return pre_start, pre_len, post_start, post_len, total_delta

def extract_minimal_patch(model_patch):
    """
    Wrapper function that takes hunk and
    * Removes trailing non +/- lines and trailing whitespace per line per hunk
    * Recalculates hunk start/end position and diff delta
    * Returns new patch
    """
    model_patch = model_patch.lstrip("\n")
    new_patch = ""
    for patch in PATCH_PATTERN.findall(model_patch):
        total_delta = 0
        patch_header = PATCH_FILE_PATTERN.findall(patch)[0]
        if patch_header:
            new_patch += patch_header + "\n"
        for hunk in PATCH_HUNK_PATTERN.findall(patch):
            pre_start, pre_len, post_start, post_len, content = hunk
            pre_start, pre_len, post_start, post_len, content = list(
                map(lambda x: int(x) if x.isnumeric() else x, hunk)
            )
            content = strip_comment_lines(content)
            content, adjust_pre_start = strip_content(content)
            pre_start += adjust_pre_start
            pre_start, pre_len, post_start, post_len, total_delta = get_hunk_stats(
                pre_start, pre_len, post_start, post_len, content, total_delta
            )
            new_patch += (
                f"@@ -{pre_start},{pre_len} +{post_start},{post_len} @@{content}"
            )
    return new_patch


# adapted from https://github.com/R2E-Gym/R2E-Gym/tree/main/src/r2egym/commit_models
class EntityType(Enum):
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    STATEMENT = "statement"
    IMPORT = "import"


class Entity(BaseModel):
    file_name: Path
    type: EntityType
    name: str
    content: str
    ast_type_str: str
    start_lineno: int
    end_lineno: int
    parent: Optional["Entity"] = None

    def __hash__(self) -> int:
        if self.type in (EntityType.FUNCTION, EntityType.CLASS, EntityType.METHOD):
            return hash((self.file_name, self.type, self.name))
        else:
            return hash((self.file_name, self.type, self.content))

    def __lt__(self, other: "Entity") -> bool:
        return self.start_lineno < other.start_lineno

    def __eq__(self, other: "Entity") -> bool:
        if self.file_name != other.file_name:
            return False
        if self.type in (EntityType.FUNCTION, EntityType.CLASS, EntityType.METHOD):
            return self.name == other.name
        return self.content == other.content

    def prompt_repr(self) -> str:
        if self.type in (EntityType.FUNCTION, EntityType.CLASS, EntityType.METHOD):
            return f"{self.type.value} '{self.name}' -- {self.file_name}:{self.start_lineno}-{self.end_lineno}"

    def json_summary_dict(self) -> dict:
        return {
            "file_name": str(self.file_name),
            "type": self.type.value,
            "name": self.name,
            "ast_type_str": self.ast_type_str,
            "start_lineno": self.start_lineno,
            "end_lineno": self.end_lineno,
        }


class Range(BaseModel):
    start: int
    length: int | None = None

    def get_patch(self) -> str:
        if self.length is None:
            return f"{self.start}"
        return f"{self.start},{self.length}"


class UnitHunkDescriptor(BaseModel):
    old_range: Range
    new_range: Range
    section: str

    def get_patch(self) -> str:
        content = f"@@ -{self.old_range.get_patch()} +{self.new_range.get_patch()} @@"
        if self.section:
            content += f" {self.section}"
        return content


class LineType(Enum):
    CONTEXT = "context"
    ADDED = "added"
    DELETED = "deleted"
    NOTE = "note"


class Line(BaseModel):
    content: str
    type: LineType

class LineGroup(BaseModel):
    all_lines: list[Line] = Field(default_factory=list)

    @property
    def num_deleted(self) -> int:
        return sum(line.type == LineType.DELETED for line in self.all_lines)

    @property
    def num_added(self) -> int:
        return sum(line.type == LineType.ADDED for line in self.all_lines)

    @property
    def num_context(self) -> int:
        return sum(line.type == LineType.CONTEXT for line in self.all_lines)

    @property
    def lr_lines(self) -> list[Line]:
        return [
            line
            for line in self.all_lines
            if line.type in [LineType.DELETED, LineType.CONTEXT]
        ]

    @property
    def num_edited(self) -> int:
        return self.num_deleted + self.num_added


class UniHunk(BaseModel):
    descriptor: UnitHunkDescriptor
    line_group: LineGroup
    modified_entities: set[Entity] = Field(default_factory=set)
    added_entities: set[Entity] = Field(default_factory=set)
    deleted_entities: set[Entity] = Field(default_factory=set)

    @property
    def is_import_hunk(self) -> bool:
        for line in self.line_group.lr_lines:
            if len(line.content.strip()) == 0:
                continue
            if line.content.startswith("import"):
                continue
            if line.content.startswith("from ") and "import" in line.content:
                continue
            return False
        return True

    @property
    def is_insert_hunk(self) -> bool:
        return self.line_group.num_deleted == 0

    @property
    def is_delete_hunk(self) -> bool:
        return self.line_group.num_added == 0

    @property
    def edited_entities(self) -> set[Entity]:
        return self.modified_entities.union(self.added_entities).union(
            self.deleted_entities
        )

    @property
    def num_edited_entities(self) -> int:
        return len(self.edited_entities)

    @property
    def num_modified_entities(self) -> int:
        return len(self.modified_entities)

    @property
    def num_added_entities(self) -> int:
        return len(self.added_entities)

    @property
    def num_deleted_entities(self) -> int:
        return len(self.deleted_entities)

    @property
    def num_method_entities(self) -> int:
        return sum(entity.type == EntityType.METHOD for entity in self.edited_entities)

    @property
    def num_function_entities(self) -> int:
        return sum(
            entity.type == EntityType.FUNCTION for entity in self.edited_entities
        )

    @property
    def num_class_entities(self) -> int:
        return sum(entity.type == EntityType.CLASS for entity in self.edited_entities)

    @property
    def edit_transcends_single_location(self) -> bool:
        return (self.num_function_entities + self.num_class_entities > 1) or (
            self.num_method_entities > 1
        )


class FileInfo(BaseModel):
    path: str


class FileDiffHeader(BaseModel):
    file: FileInfo
    misc_line: str | None = None

    @property
    def path(self) -> str:
        return self.file.path

    @property
    def is_test_file(self) -> bool:
        return (
            self.path.endswith("_test.py")
            or self.path.startswith("test_")
            or "tests" in self.path.split("/")
        )

    def get_patch(self) -> str:
        patch = f"diff --git a/{self.file.path} b/{self.file.path}\n"
        if self.misc_line:
            patch += self.misc_line + "\n"
        return patch


class IndexLine(BaseModel):
    old_commit_hash: str
    new_commit_hash: str
    mode: str

    def get_patch(self) -> str:
        return f"index {self.old_commit_hash}..{self.new_commit_hash}{' ' if self.mode else ''}{self.mode}\n"


class FileDiff(BaseModel):
    old_file_content: str
    new_file_content: str
    header: FileDiffHeader
    index_line: IndexLine | None = None
    is_binary_file: bool = False
    binary_line: str | None = None
    minus_file: FileInfo | None = None
    plus_file: FileInfo | None = None
    hunks: list[UniHunk] = []

    @property
    def path(self) -> str:
        return self.header.path

    @property
    def is_test_file(self) -> bool:
        return (
            self.path.endswith("_test.py")
            or self.path.startswith("test_")
            or self.path.split("/")[-1].startswith("test_")
            or "tests" in self.path.split("/")
            or "Tests" in self.path.split("/")
            or "test" in self.path.split("/")
            or "Test" in self.path.split("/")
        )

    @property
    def is_mypy_test_file(self) -> bool:
        return self.path.endswith(".test")

    def get_patch(self) -> str:
        patch = self.header.get_patch()
        if self.index_line:
            patch += self.index_line.get_patch()
        if self.is_binary_file:
            patch += self.binary_line + "\n"

        if self.minus_file and self.plus_file:
            patch += f"--- {self.minus_file.path}\n"
            patch += f"+++ {self.plus_file.path}\n"
        for hunk in self.hunks:
            patch += hunk.descriptor.get_patch() + "\n"
            for line in hunk.line_group.all_lines:
                if line.type == LineType.CONTEXT:
                    patch += f" {line.content}\n"
                elif line.type == LineType.ADDED:
                    patch += f"+{line.content}\n"
                elif line.type == LineType.DELETED:
                    patch += f"-{line.content}\n"
                elif line.type == LineType.NOTE:
                    patch += f"\\ {line.content}\n"

        return patch

    @property
    def is_python_file(self) -> bool:
        return self.path.endswith(".py")

    @property
    def num_hunks(self) -> int:
        return len(self.hunks)

    @property
    def num_edited_lines(self) -> int:
        return sum(hunk.line_group.num_edited for hunk in self.hunks)

    @property
    def edited_entities(self) -> set[Entity]:
        return {entity for hunk in self.hunks for entity in hunk.edited_entities}

    @property
    def added_entities(self) -> set[Entity]:
        return {entity for hunk in self.hunks for entity in hunk.added_entities}

    @property
    def deleted_entities(self) -> set[Entity]:
        return {entity for hunk in self.hunks for entity in hunk.deleted_entities}

    @property
    def modified_entities(self) -> set[Entity]:
        return {entity for hunk in self.hunks for entity in hunk.modified_entities}

    @property
    def num_edited_entities(self) -> int:
        return len(self.edited_entities)

    @property
    def num_added_entities(self) -> int:
        return len(self.added_entities)

    @property
    def num_deleted_entities(self) -> int:
        return len(self.deleted_entities)

    @property
    def num_modified_entities(self) -> int:
        return len(self.modified_entities)

    @property
    def num_method_entities(self) -> int:
        return sum(entity.type == EntityType.METHOD for entity in self.edited_entities)

    @property
    def num_function_entities(self) -> int:
        return sum(
            entity.type == EntityType.FUNCTION for entity in self.edited_entities
        )

    @property
    def num_class_entities(self) -> int:
        return sum(entity.type == EntityType.CLASS for entity in self.edited_entities)

    @property
    def is_new(self) -> bool:
        return self.old_file_content == "/dev/null" or self.old_file_content == ""

    def generate_hunks_from_content(self) -> None:
        if not self.old_file_content or not self.new_file_content:
            return
        
        old_lines = self.old_file_content.splitlines()
        new_lines = self.new_file_content.splitlines()
        
        diff = list(cydifflib.unified_diff(
            old_lines, 
            new_lines, 
            fromfile=f"a/{self.path}",
            tofile=f"b/{self.path}",
            n=3  # context lines
        ))
        
        hunks = []
        current_hunk_lines = []
        current_descriptor = None
        
        i = 0
        while i < len(diff):
            line = diff[i]
            
            # look for hunk header (e.g., "@@ -1,4 +1,4 @@")
            if line.startswith("@@"):
                # save previous hunk if exists
                if current_descriptor and current_hunk_lines:
                    line_group = LineGroup(all_lines=current_hunk_lines)
                    hunks.append(UniHunk(descriptor=current_descriptor, line_group=line_group))
                
                current_descriptor = self._parse_hunk_header(line)
                current_hunk_lines = []
                
            elif line.startswith(" "):
                current_hunk_lines.append(Line(content=line[1:], type=LineType.CONTEXT))
            elif line.startswith("-"):
                current_hunk_lines.append(Line(content=line[1:], type=LineType.DELETED))
            elif line.startswith("+"):
                current_hunk_lines.append(Line(content=line[1:], type=LineType.ADDED))
            elif line.startswith("\\"):
                # note line (e.g., "\ No newline at end of file")
                current_hunk_lines.append(Line(content=line[2:], type=LineType.NOTE))
            
            i += 1
        
        # add final hunk
        if current_descriptor and current_hunk_lines:
            line_group = LineGroup(all_lines=current_hunk_lines)
            hunks.append(UniHunk(descriptor=current_descriptor, line_group=line_group))
        
        self.hunks = hunks
    
    def _parse_hunk_header(self, header_line: str) -> UnitHunkDescriptor:
        header_content = header_line.strip()
        if header_content.startswith("@@"):
            header_content = header_content[2:]
        if header_content.endswith("@@"):
            header_content = header_content[:-2]
        
        # split on space to separate ranges from (optional) function context
        parts = header_content.strip().split(" ", 2)
        old_range_str = parts[0]  # e.g., "-1,4"
        new_range_str = parts[1]  # e.g., "+1,4"
        section = parts[2] if len(parts) > 2 else ""
        
        old_range = self._parse_range(old_range_str[1:]) 
        new_range = self._parse_range(new_range_str[1:]) 
        
        return UnitHunkDescriptor(
            old_range=old_range,
            new_range=new_range,
            section=section
        )
    
    def _parse_range(self, range_str: str) -> Range:
        if "," in range_str:
            start, length = range_str.split(",")
            return Range(start=int(start), length=int(length))
        else:
            return Range(start=int(range_str), length=None)
