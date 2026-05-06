# atop Rawlog Compatibility Matrix

atop-web parses `atop` rawlog files directly. This document describes which
atop versions and host operating systems are supported, how dispatch works
internally, and how to validate a new rawlog source against the parser.

## Supported atop versions

| atop version | Status        | Shared layout with | Notes                                                  |
| ------------ | ------------- | ------------------ | ------------------------------------------------------ |
| 2.7.x        | Supported     | -                  | Pre-cgroup record; no `availablemem`; no `inflight`.   |
| 2.10.x       | Supported     | -                  | Ubuntu 24.04 default; distinct struct sizes; pre-cgroup rawrecord. |
| 2.11.x       | Supported     | 2.12 (CDEF shared) | Struct-level identical to 2.12; dispatched separately. |
| 2.12.x       | Supported     | -                  | Reference layout (AL2023 / Ubuntu 26.04).              |
| < 2.7        | Not supported | -                  | Out of scope.                                          |
| 2.8, 2.9     | Not tested    | -                  | No sample captures. File an issue with a rawlog.       |

## OS compatibility matrix

Measured against in-house captures on 2026-05-06. `tstat` and `sstat`
columns are the byte sizes the parser reads from `rawheader.tstatlen` /
`rawheader.sstatlen`.

| OS                 | atop   | tstat | sstat     | Status                       |
| ------------------ | ------ | ----- | --------- | ---------------------------- |
| RHEL 8.10          | 2.7.1  | 840   | 954,360   | Supported (via SPEC_2_7)     |
| RHEL 9.7           | 2.7.1  | 840   | 954,360   | Supported (via SPEC_2_7)     |
| RHEL 10.1          | 2.11.1 | 968   | 1,064,016 | Supported (via SPEC_2_11)    |
| Ubuntu 24.04       | 2.10.0 | 992   | 1,030,216 | Supported (via SPEC_2_10)    |
| Ubuntu 26.04       | 2.12.1 | 968   | 1,064,016 | Supported (via SPEC_2_12)    |
| SLES 15-SP7        | 2.11.1 | 968   | 1,064,016 | Supported (via SPEC_2_11)    |
| SLES 16.0          | 2.11.1 | 968   | 1,064,016 | Supported (via SPEC_2_11)    |
| Amazon Linux 2     | 2.7.1  | 840   | 954,360   | Supported (via SPEC_2_7)     |
| Amazon Linux 2023  | 2.12.x | 968   | 1,064,016 | Supported (via SPEC_2_12)    |

## Dispatch internals

`atop_web/parser/reader.py` keeps one `VersionSpec` per atop revision and
looks them up by the triple `(tstatlen, sstatlen, aversion)` from the
rawheader. `aversion` is the atop version encoded as
`((major & 0x7f) << 8) | (minor & 0xff)`:

| atop   | aversion |
| ------ | -------- |
| 2.7    | 0x8207   |
| 2.10   | 0x820A   |
| 2.11   | 0x820B   |
| 2.12   | 0x820C   |

`VERSION_TABLE` in `reader.py` registers both exact matches and an
`aversion=None` fallback for each known struct layout. If a future
atop 2.13 keeps the 2.12 struct layout, the fallback decodes it as 2.12;
the moment 2.13 diverges, the fallback must be removed and an explicit
entry added.

atop 2.11 and 2.12 share the identical on-disk layout (same tstat size,
same sstat offsets), so `SPEC_2_11.cdef_filename` is `atop_2_12.cdef`.
The two specs remain distinct instances so operator-facing diagnostics
(`RawLog.spec.name`, unsupported-version error messages) report the true
atop major.minor of the capture.

## Validating a new atop version

Before enabling a new atop revision in the parser, confirm the struct
identity matches something already supported. Run the following snippet
against a capture produced by the new atop build:

```python
import struct
from pathlib import Path

with Path("/var/log/atop/NEW_RAWLOG").open("rb") as fh:
    buf = fh.read(36)

magic    = struct.unpack_from("<I", buf, 0)[0]
aversion = struct.unpack_from("<H", buf, 4)[0]
rawhdrln = struct.unpack_from("<H", buf, 10)[0]
sstatlen = struct.unpack_from("<I", buf, 28)[0]
tstatlen = struct.unpack_from("<I", buf, 32)[0]

major = (aversion >> 8) & 0x7F
minor = aversion & 0xFF
print(f"magic=0x{magic:08x} atop={major}.{minor} "
      f"rawheadlen={rawhdrln} tstat={tstatlen} sstat={sstatlen}")
```

Expected outputs for known versions:

```
atop 2.7.x : magic=0xfeedbeef atop=2.7  rawheadlen=480 tstat=840  sstat=954360
atop 2.10.x: magic=0xfeedbeef atop=2.10 rawheadlen=480 tstat=992  sstat=1030216
atop 2.11.x: magic=0xfeedbeef atop=2.11 rawheadlen=480 tstat=968  sstat=1064016
atop 2.12.x: magic=0xfeedbeef atop=2.12 rawheadlen=480 tstat=968  sstat=1064016
```

If `tstat` and `sstat` match a supported pair and the atop version is
close (same major, later minor), the parser's `aversion=None` fallback
will decode it. Otherwise the struct layout has changed and requires:

1. A new CDEF in `atop_web/parser/layouts/`.
2. A new `SPEC_...` in `reader.py` with offsets measured against a real
   rawlog written by that atop build (not `sizeof` on the host).
3. Explicit entries in `VERSION_TABLE` keyed on the new `(tstatlen,
   sstatlen, aversion)` triple.
4. A test fixture and at least one parsing test against the new capture.

## Known gaps

- **atop 2.8 / 2.9**: no capture samples in the test corpus; behavior
  undefined. Report an issue with a capture attached if you need support.
- **Cross-architecture rawlogs**: all tested captures are x86_64. ARM /
  ppc captures have not been validated; struct padding may differ and
  will need a separate `SPEC_...` variant.
