# Regex Rules Reference

All regex patterns used in `src/pii/run.py` for rule-based PII detection.

---

## PHONE

### `PHONE_RE` — Digit phone numbers

```python
re.compile(r"(?<!\d)(?:\+?\d[\d\-\s]{6,}\d)(?!\d)")
```

| Component | Meaning |
|-----------|---------|
| `(?<!\d)` | Not preceded by a digit |
| `\+?` | Optional `+` prefix (international) |
| `\d` | First digit |
| `[\d\-\s]{6,}` | 6+ digits, hyphens, or spaces |
| `\d` | Last digit |
| `(?!\d)` | Not followed by a digit |

**Minimum:** 8 characters total (1 + 6 + 1)

**Examples:** `99999999`, `9999 9999`, `+65 9123 4567`

**Config:** `pii.rules.phone` (default: `true`)

---

### `ZH_PHONE_RE` — Chinese numeral phones

```python
_ZH_DIGIT = "[零一二三四五六七八九〇○]"

re.compile(
    rf"(?:{_ZH_DIGIT}{{4}}[，、,\s]{_ZH_DIGIT}{{4}})"
    rf"|(?:{_ZH_DIGIT}{{8,}})"
)
```

**Pattern 1:** 4 Chinese digits + separator (，、, or space) + 4 Chinese digits

**Pattern 2:** 8+ consecutive Chinese digits

**Examples:** `九三四五，六六七七`, `九三四五六六七七`

**Config:** `pii.rules.zh_phone` (default: `true`)

---

### `EN_SPOKEN_PHONE_RE` — English spoken-form phones

```python
_EN_DIGIT_WORD = r"(?:zero|one|two|three|four|five|six|seven|eight|nine)"

re.compile(
    rf"(?:{_EN_DIGIT_WORD}(?:\s+{_EN_DIGIT_WORD}){{7,}})",
    re.IGNORECASE
)
```

**Matches:** 8+ consecutive spoken digit words separated by spaces.

**Example:** `nine one three two five six seven eight`

**Config:** `pii.rules.en_spoken_phone` (default: `true`)

---

### `EN_PARTIAL_PHONE_RE` — English cue-word-gated partial phone

```python
re.compile(
    r"(?:phone|number|call|dial|contact|hp|handphone)"
    r"[\s:]*"
    r"(\d{4,})",
    re.IGNORECASE
)
```

**Cue words:** phone, number, call, dial, contact, hp, handphone

**Captures:** Group 1 — 4+ digit sequence after cue word.

**Example:** `phone 9810` → captures `9810`

**Config:** `pii.rules.partial_phone` (default: `true`)

---

### `ZH_PARTIAL_PHONE_RE` — Chinese cue-word-gated partial phone

```python
_ZH_DIGIT = "[零一二三四五六七八九〇○]"

re.compile(
    r"(?:电话|手机|联系|号码|拨打)"
    r"[是为：:]*\s*"
    rf"({_ZH_DIGIT}{{3,}})"
)
```

**Cue words:** 电话, 手机, 联系, 号码, 拨打

**Captures:** Group 1 — 3+ Chinese digit characters after cue word.

**Example:** `电话是九八一零` → captures `九八一零`

**Config:** `pii.rules.partial_phone` (default: `true`)

---

## EMAIL

### `EMAIL_RE` — Email addresses

```python
re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
```

**Matches:** Standard email format — local part + `@` + domain + TLD (2+ letters).

**Example:** `user@example.com`, `john.doe+work@company.co.sg`

**Config:** `pii.rules.email` (default: `true`)

---

## ADDRESS

### `ZH_ADDRESS_RE` — Chinese addresses

```python
_ZH_DIGIT = "[零一二三四五六七八九〇○]"
_ROAD_SUFFIX = r"(?:Road|Street|Avenue|Drive|Lane|View\s+Road)"

re.compile(
    # Pattern 1: 大牌 + numerals + optional road name
    rf"(?:[\u4e00-\u9fff]{{0,6}})?大牌{_ZH_DIGIT}{{2,}}"
    rf"(?:[，、,\s]*门牌{_ZH_DIGIT}+)?"
    rf"(?:[，、,\s]*[\u4e00-\u9fff]{{2,6}}(?:路|街|道|桥))?"
    rf"(?:[，、,\s]+[A-Za-z][\w\s]*?{_ROAD_SUFFIX})?"
    # Pattern 2: Road name + optional 大牌
    rf"|[\u4e00-\u9fff]{{2,6}}(?:路|街|道|园|苑|花园|公寓|Residence)"
    rf"(?:[，、,.\s]*大牌{_ZH_DIGIT}{{2,}})?"
)
```

**Pattern 1:** `大牌` (block) + Chinese numeral block number, optionally followed by 门牌 (unit), Chinese road name (路/街/道/桥), or English road name.

**Pattern 2:** Chinese road/place name ending with 路/街/道/园/苑/花园/公寓/Residence, optionally followed by 大牌 + block number.

**Examples:** `大牌六六三`, `南京路`, `绿园路大牌六六三`

**Config:** `pii.rules.address` (default: `true`)

---

### `EN_ADDRESS_RE` — English addresses

```python
_ROAD_TYPE = r"(?:Road|Street|Avenue|Drive|Lane|Way|Crescent|Place|Boulevard|View\s+Road|Grove\s+Road)"

re.compile(
    # Alt 1: Block/Blk + number + road-type
    rf"(?:Block|Blk)\s+\d+[\s,]*[\w\s]*?{_ROAD_TYPE}"
    rf"|"
    # Alt 2: number + street name + road-type
    rf"\d{{1,4}}\s+[\w\s]*?{_ROAD_TYPE}(?:\s+\d{{1,4}})?"
    rf"|"
    # Alt 3: street name (1-2 words) + road-type + number
    rf"[\w]+(?:\s+[\w]+){{0,1}}\s+{_ROAD_TYPE}\s+\d{{1,4}}",
    re.IGNORECASE
)
```

**Road types recognized:** Road, Street, Avenue, Drive, Lane, Way, Crescent, Place, Boulevard, View Road, Grove Road

| Alternative | Format | Example |
|-------------|--------|---------|
| 1 | `Block/Blk` + number + road-type | Block 123 Orchard Road |
| 2 | number + words + road-type | 10 Orchard Road |
| 3 | words + road-type + number | Jurong East Street 12 |

**Config:** `pii.rules.address` (default: `true`)

---

### `SG_POSTAL_RE` — Singapore postal codes (prefixed)

```python
re.compile(r"(?:Singapore\s*|S)\d{6}(?!\d)", re.IGNORECASE)
```

| Component | Meaning |
|-----------|---------|
| `(?:Singapore\s*\|S)` | Prefix: "Singapore" (with optional space) or "S" |
| `\d{6}` | Exactly 6 digits |
| `(?!\d)` | Not followed by another digit |

**Examples:** `S609690`, `Singapore 123456`, `Singapore609690`

**Config:** `pii.rules.postal_code` (default: `true`)

---

### `SG_POSTAL_AFTER_ADDR_RE` — Bare 6-digit postal codes after address

```python
_ROAD_TYPE = r"(?:Road|Street|Avenue|Drive|Lane|Way|Crescent|Place|Boulevard|View\s+Road|Grove\s+Road)"

re.compile(
    rf"{_ROAD_TYPE}"
    r"(?:\s+\d{1,4})?"     # optional street number
    r"[,\s]+"               # separator (comma, space)
    r"(\d{6})(?!\d)",
    re.IGNORECASE,
)
```

| Component | Meaning |
|-----------|---------|
| `_ROAD_TYPE` | Road type keyword (Road, Street, Avenue, etc.) |
| `(?:\s+\d{1,4})?` | Optional street number after road type |
| `[,\s]+` | Comma and/or space separator |
| `(\d{6})` | **Capture group 1** — exactly 6 digits (the postal code) |
| `(?!\d)` | Not followed by another digit |

**Examples:**
- `Jurong East Street 12, 609690` → captures `609690`
- `Orchard Road, 238879` → captures `238879`
- `Block 123 Tampines Street 45 520123` → captures `520123`

**Config:** `pii.rules.postal_code` (default: `true`)

---

## ID

### `SG_NRIC_RE` — Singapore NRIC/FIN numbers

```python
re.compile(r"(?<![A-Za-z])[STFGM]\d{7}[A-Za-z](?![A-Za-z])", re.IGNORECASE)
```

| Component | Meaning |
|-----------|---------|
| `(?<![A-Za-z])` | Not preceded by a letter |
| `[STFGM]` | Prefix letter: S/T (citizens), F/G (foreigners), M (post-2022 foreigners) |
| `\d{7}` | 7-digit serial number |
| `[A-Za-z]` | Check letter |
| `(?![A-Za-z])` | Not followed by a letter |

**Examples:** `S1234567A`, `T0123456J`, `G7654321B`, `M1234567K`

**Config:** `pii.rules.nric` (default: `true`)

---

## Config Toggle Summary

| Toggle | Default | Patterns |
|--------|---------|----------|
| `pii.rules.phone` | `true` | `PHONE_RE` |
| `pii.rules.zh_phone` | `true` | `ZH_PHONE_RE` |
| `pii.rules.en_spoken_phone` | `true` | `EN_SPOKEN_PHONE_RE` |
| `pii.rules.partial_phone` | `true` | `EN_PARTIAL_PHONE_RE`, `ZH_PARTIAL_PHONE_RE` |
| `pii.rules.email` | `true` | `EMAIL_RE` |
| `pii.rules.address` | `true` | `ZH_ADDRESS_RE`, `EN_ADDRESS_RE` |
| `pii.rules.postal_code` | `true` | `SG_POSTAL_RE`, `SG_POSTAL_AFTER_ADDR_RE` |
| `pii.rules.nric` | `true` | `SG_NRIC_RE` |
