"""
Tests for the Tigrinya rule-based stemmer.
All examples marked [NS] confirmed directly by a native Tigrinya speaker.
Supporting sources: [3] COLING 2012, [4] LING073 Swarthmore
"""
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nlp_pipeline import _stem_tigrinya_word, _lemmatize_tigrinya, _TIGRINYA_SUFFIXES

def test_stemmer():
    tests = []
    fails = []
    def check(name, got, expected):
        if got == expected: 
            tests.append(f"✅  {name}")
        else: 
            fails.append(f"❌  {name}\n     got={repr(got)}  expected={repr(expected)}")

    # ── Noun plurals ──────────────────────────────────────────────────────
    check("ከባቢታት → ከባቢ  (-ታት plural) [NS,Src3,4]",  _stem_tigrinya_word("ከባቢታት"),  "ከባቢ")
    check("ጻዕርታት → ጻዕር  (-ታት plural) [NS,Src4]",    _stem_tigrinya_word("ጻዕርታት"),  "ጻዕር")
    check("ተማሃሪት → ተማሃሪ (-ት feminine) [NS]",         _stem_tigrinya_word("ተማሃሪት"),  "ተማሃሪ")

    # ── 1sg object 'me' ───────────────────────────────────────────────────
    check("ፈሊጡኒ → ፈሊጡ  (-ኒ 'me') [NS]",              _stem_tigrinya_word("ፈሊጡኒ"),   "ፈሊጡ")

    # ── 2sg.m object 'you(m)' — both spellings ───────────────────────────
    check("ፈሊጡካ → ፈሊጡ  (-ካ 'you(m)' regular) [NS]",  _stem_tigrinya_word("ፈሊጡካ"),   "ፈሊጡ")
    check("ፈሊጡኻ → ፈሊጡ  (-ኻ 'you(m)' labiovelar) [NS]",_stem_tigrinya_word("ፈሊጡኻ"),   "ፈሊጡ")

    # ── 2sg.f object 'you(f)' — both spellings ───────────────────────────
    check("ኣሎኺ  → ኣሎ   (-ኺ 'you(f)' labiovelar) [NS]", _stem_tigrinya_word("ኣሎኺ"),   "ኣሎ")
    check("ኣሎኪ  → ኣሎ   (-ኪ 'you(f)' regular) [NS]",    _stem_tigrinya_word("ኣሎኪ"),   "ኣሎ")

    # ── 1pl object 'us' ───────────────────────────────────────────────────
    check("ፈሊጡና → ፈሊጡ  (-ና 'us') [NS]",               _stem_tigrinya_word("ፈሊጡና"),   "ፈሊጡ")

    # ── 2pl.m / 2pl.f ────────────────────────────────────────────────────
    check("ኣሎኹም → ኣሎ   (-ኹም 'you(pl.m)') [NS,Src4]",  _stem_tigrinya_word("ኣሎኹም"),   "ኣሎ")
    check("ኣሎኽን → ኣሎ   (-ኽን 'you(pl.f)') [NS,Src4]",  _stem_tigrinya_word("ኣሎኽን"),   "ኣሎ")

    # ── 3sg.m object 'him' — three subject-conditioned forms ─────────────
    check("ፈሊጠዮ → ፈሊጠ  (-ዮ 'him' after I/you stem) [NS]", _stem_tigrinya_word("ፈሊጠዮ"), "ፈሊጠ")
    check("ፈሊጡዎ → ፈሊጡ  (-ዎ 'him' after he/they stem) [NS]",_stem_tigrinya_word("ፈሊጡዎ"), "ፈሊጡ")
    check("ፈሊጣቶ → ፈሊጣ  (-ቶ 'him' after she stem) [NS]",    _stem_tigrinya_word("ፈሊጣቶ"), "ፈሊጣ")
    check("ፈሊጥናዮ→ ፈሊጥና (-ዮ after we-stem) [NS]",           _stem_tigrinya_word("ፈሊጥናዮ"),"ፈሊጥና")
    check("ፈሊጠምዎ→ ፈሊጠም (-ዎ after they-stem) [NS]",         _stem_tigrinya_word("ፈሊጠምዎ"),"ፈሊጠም")

    # ── 3sg.f object 'her' ───────────────────────────────────────────────
    check("ፈሊጡዋ → ፈሊጡ  (-ዋ 'her') [NS]",               _stem_tigrinya_word("ፈሊጡዋ"),   "ፈሊጡ")

    # ── 3pl.m object 'them(m)' ───────────────────────────────────────────
    check("ፈሊጡዎም→ ፈሊጡ  (-ዎም 'them(m)') [NS]",           _stem_tigrinya_word("ፈሊጡዎም"),  "ፈሊጡ")

    # ── Possessive 1sg 'my' ───────────────────────────────────────────────
    check("ከልበይ → ከልበ  (-ይ 'my') [NS]",                 _stem_tigrinya_word("ከልበይ"),   "ከልበ")

    # ── Longest-first ordering ────────────────────────────────────────────
    check("ፈሊጡዎም strips -ዎም not -ዎ",                    _stem_tigrinya_word("ፈሊጡዎም"),  "ፈሊጡ")
    check("ኣሎኹም  strips -ኹም not -ካ",                    _stem_tigrinya_word("ኣሎኹም"),   "ኣሎ")
    lengths = [len(s) for s, _ in _TIGRINYA_SUFFIXES]
    check("Suffix list ordered longest-first",
          all(lengths[i] >= lengths[i+1] for i in range(len(lengths)-1)), True)

    # ── No-strip edge cases ───────────────────────────────────────────────
    check("ሰብ   → unchanged (no matching suffix)",        _stem_tigrinya_word("ሰብ"),     "ሰብ")
    check("ካ    → unchanged (word IS the suffix)",        _stem_tigrinya_word("ካ"),      "ካ")
    check("ዎም   → unchanged (word IS the suffix)",        _stem_tigrinya_word("ዎም"),     "ዎም")
    check("''   → ''  (empty string)",                    _stem_tigrinya_word(""),       "")
    check("hello→ hello (Latin passthrough)",              _stem_tigrinya_word("hello"),  "hello")

    # ── Batch ─────────────────────────────────────────────────────────────
    check("Batch: mixed NS words",
        _lemmatize_tigrinya(["ፈሊጡኒ","ፈሊጡዎም","ከባቢታት","ፈሊጣቶ","ሰብ"]),
        ["ፈሊጡ","ፈሊጡ","ከባቢ","ፈሊጣ","ሰብ"])
    check("Batch: empty list",      _lemmatize_tigrinya([]), [])
    check("Batch: 1-to-1 length",   len(_lemmatize_tigrinya(["ፈሊጡኒ","ኣሎኹም","ሰብ"])), 3)
    f = _lemmatize_tigrinya(["ፈሊጡኒ","ከባቢታት"])
    check("Batch: idempotent",      f, _lemmatize_tigrinya(f))
    check("Batch: no-match passthrough",
        _lemmatize_tigrinya(["ሰብ","ሃገር"]), ["ሰብ","ሃገር"])
    check("Batch: English passthrough",
        _lemmatize_tigrinya(["running","foxes"]), ["running","foxes"])

    # ── Report ────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  TIGRINYA STEMMER — NATIVE SPEAKER VERIFIED ({len(tests)+len(fails)} tests)")
    print(f"{'='*62}")
    for t in tests: print(t)
    for f in fails: print(f)
    print(f"{'='*62}")
    print(f"  PASSED: {len(tests)}   FAILED: {len(fails)}")
    print(f"  {'✅ ALL PASSED' if not fails else f'❌ {len(fails)} FAILURES'}")
    
    assert len(fails) == 0

if __name__ == "__main__":
    test_stemmer()
