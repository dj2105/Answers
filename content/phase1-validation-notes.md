# Phase 1 Validation Notes

## Explicit ambiguities and date uncertainties

1. **“Initial email to HR” has no explicit date in the supplied page text.**
   - This is the clearest date uncertainty in the provided Phase 1 material.
   - It must not be assigned a fabricated `date_iso` value in later build stages.
   - It should remain marked as **unconfirmed** or visually treated as an undated/floating item.

2. **The context paragraph contains a dated statement (“On 18 August 2025 I moved to a team...”) but the context block itself is not a dated event card.**
   - Later UI stages should not convert the whole context paragraph into a normal timeline event unless separately instructed.

3. **The 11 Nov handoff item is outside the formal Phase 1 date range.**
   - It is visible on the page and should be captured.
   - It should not be folded into the formal “18 Aug — 31 Oct 2025” range as though it were an in-range Phase 1 event.

## Layout-implied relationships that are not fully textual

1. **Some items may be visually grouped by lane or proximity rather than explicit textual linkage.**
   - The supplied text indicates separate blocks, but not a fully specified screen geometry.
   - Later UI work may cluster related items visually, but must not imply stronger causal or dated relationships than the source states.

2. **“Initial email to HR” appears both as a standalone undated Phase 1 item and in the visible continuation/handoff text.**
   - This implies cross-phase relevance.
   - It does **not** confirm a specific sent date for the initial email within the Phase 1 timeline.

3. **The “SENT” and “RECEIVED” chips imply procedural progression.**
   - The layout may connect them visually.
   - The source text does not explicitly narrate the full intermediate chain on this page, so later stages must avoid filling gaps.

## Wording that should remain unchanged

The following wording may sound stylistically uneven, but should remain unchanged because the brief requires documentary fidelity:

- “The cases above highlight different Ofcom regulatory concerns that arose while handling customer issues.”
- “Ofcom requires complaints remain open until the customer considers it resolved.”
- “I … understand that you want this complaint closed, but ... part of my responsibility as an agent is to be aware of Ofcom's regulations.”
- “offering no resolution, and closing the complaint without permission - preventing access to ADR.”

Specific preservation notes:

- Keep **Sept** and **Oct** exactly as given in the source text where used in date labels.
- Keep the square-bracketed interpolation in `"[I’ve] only received very vague guidance..."`.
- Keep the ellipsis forms exactly as supplied: one entry uses the ellipsis character (`…`), and another uses three full stops (`...`).
- Keep the hyphen in `permission - preventing access to ADR` rather than silently converting it to an em dash.
- Keep the apostrophe in `Ofcom's` as supplied.

## Things that later build stages must NOT guess

1. **Do not invent a date for “Initial email to HR”.**
2. **Do not infer exact x-axis placement for undated material as if it were confirmed chronology.**
3. **Do not deduplicate T-22 across 6 Oct and 20 Oct.**
   - The supplied page uses the same evidence code in two separate content items.
4. **Do not rewrite titles or standardise them into cleaner UI copy.**
   - The wording should remain evidential and documentary.
5. **Do not infer missing narrative steps between GH-01 SENT and GH-04 RECEIVED.**
6. **Do not infer that every item with Ofcom commentary has the same card style or evidential weight.**
   - Operational cases, internal messages, training references, and status chips are distinct content types in the source.
7. **Do not treat the context paragraph or evidence legend as dated events by default.**

## Implementation caution

If later UI stages need a single chronological ordering for all objects, they should use:

- exact dates for confirmed items,
- explicit `unconfirmed` handling for undated items,
- and a visually separate continuation zone for the 11 Nov handoff.

This is necessary to preserve chronology integrity without overstating what the page explicitly confirms.
