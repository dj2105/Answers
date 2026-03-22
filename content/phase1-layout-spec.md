# Phase 1 Layout Spec

## 1. Displayed date range

- **Primary displayed range:** 18 Aug — 31 Oct 2025
- **Range start (exact):** 2025-08-18
- **Range end (exact):** 2025-10-31
- **Visible continuation beyond range:** 11 Nov 2025 for the Phase 2 handoff zone only

## 2. Day counts by month

Use exact calendar spacing for date-proportional placement.

- **August 2025 portion shown:** 18 Aug — 31 Aug = 14 days
- **September 2025:** 30 days
- **October 2025:** 31 days
- **Total Phase 1 displayed span:** 75 days inclusive
- **Phase 2 handoff extension visible on Phase 1 page:** 1 Nov — 11 Nov = 11 days inclusive beyond the formal Phase 1 end date

## 3. Recommended px-per-day scale

Recommended baseline for a readable desktop build:

- **16 px per day** minimum for compact implementation
- **20 px per day** preferred for full-text readability
- At **20 px/day**, the formal Phase 1 span requires **1500 px** of horizontal date width
- If the 11 Nov handoff is shown in the same proportional axis, add **220 px** for the 1 Nov — 11 Nov extension

Recommended implementation:

- Main timeline axis: **20 px/day**
- Card width: content-driven, with min/max widths to preserve legibility
- Exact-date anchor point should align to the correct day on the axis even when card bodies extend wider than a single-day column

## 4. Month widths

At the preferred **20 px/day** scale:

- **Aug 18–31 segment:** 14 days × 20 px = **280 px**
- **September:** 30 days × 20 px = **600 px**
- **October:** 31 days × 20 px = **620 px**
- **Formal Phase 1 width:** **1500 px**
- **Optional handoff extension to 11 Nov:** 11 days × 20 px = **220 px**

If a responsive variant is needed, preserve month-to-month proportionality even if the scale is reduced.

## 5. Lane system

Use a multi-lane horizontal timeline while keeping a single shared date axis.

### Recommended lanes

1. **Header lane**
   - Phase title
   - Date range

2. **Context / legend lane**
   - Context paragraph block
   - Evidence legend block
   - These are not date-anchored events; they should sit at the left/start region of the timeline container

3. **Upper cases lane**
   - Operational case cards
   - 22 Aug T-01
   - 25 Sept T-06
   - 6 Oct T-22
   - 31 Oct T-29

4. **Mid process lane**
   - Training / procedural items with exact dates
   - 20 Oct Sec 3

5. **Lower communications lane**
   - Internal communication cards
   - 6 Sept T-09
   - 20 Oct T-22

6. **Lower process lane**
   - Undated process marker
   - Initial email to HR

7. **Status chip lane**
   - 30 Oct / SENT / GH-01

8. **Handoff lane**
   - 11 Nov / RECEIVED / GH-04 continuation marker

## 6. Items that must anchor to exact dates

These items have explicit dates in the supplied source text and should anchor to exact day positions:

- 22 Aug 2025 — T-01 — Silence Treated as Consent to Contract
- 6 Sept 2025 — T-09 — Teams DM to OPs Manager
- 25 Sept 2025 — T-06 — TL Closed Complaint Despite Customer Instruction
- 6 Oct 2025 — T-22 — Translator Refused and ADR Blocked
- 20 Oct 2025 — Sec 3 — Complaints Training
- 20 Oct 2025 — T-22 — Teams DM to TL
- 30 Oct 2025 — SENT — GH-01
- 31 Oct 2025 — T-29 — Vulnerable Customer Escalation
- 11 Nov 2025 — RECEIVED — GH-04, but only in the continuation / Phase 2 handoff zone

## 7. Items that float because their exact date is not explicit

The following item must not be pinned to an invented date:

- **Initial email to HR**
  - The supplied page text gives no explicit date for this item.
  - It should appear in a related lower/process lane as a floating panel.
  - It may be positioned visually near the 6 Sept communication cluster or toward the lead-in to the 11 Nov handoff, but it must be marked as **undated / unconfirmed**.
  - Do not generate a false point on the date axis for it.

The context paragraph and evidence legend also float outside exact chronology.

## 8. Items that are part of the Phase 2 handoff zone

These elements are visible on the Phase 1 page but belong to the continuation area rather than the formal Phase 1 range:

- PHASE 1
- Initial email to HR.
- ER acknowledges receipt
- Policy and dignity assurances given.
- 11 Nov
- RECEIVED
- GH-04

Rendering guidance:

- Place them in a visually distinct continuation block at the far right.
- Keep them attached to the shared axis if showing date proportionality through 11 Nov.
- Distinguish this zone with a continuation divider, fade, bracket, or labelled transition marker.

## 9. Distinguishing content categories visually

Maintain documentary clarity rather than decorative styling.

### Operational cases
- Larger full cards
- Strong evidence/date header line
- GC references visible in-card
- Highest visual priority

### Internal communications
- Quote-led cards
- Evidence code visible
- Slightly lighter visual weight than operational cases

### Procedural / training markers
- Medium-weight cards or panels
- Section reference or process descriptor visible
- Keep wording fully intact

### Status chips
- Small but legible date + status + evidence chip stack
- Preserve exact casing: SENT / RECEIVED

### Legend / context
- Static supporting blocks at the left/start of the timeline
- Do not style as events

### Continuation marker
- Visually separate from the formal phase span
- Must read as visible next-step / handoff material rather than a normal in-range event

## 10. Notes on chronology integrity

- Preserve the exact documentary order while also respecting exact dates.
- Do not infer a date for “Initial email to HR”.
- Do not merge the two T-22 references; they are separate visible content items with separate dates and functions.
- Do not collapse the 30 Oct SENT chip into the undated HR marker or the 11 Nov RECEIVED handoff.
- The Phase 2 handoff material may be shown on the same canvas, but it must remain clearly outside the formal Phase 1 range.
- Full text should remain visible by default; avoid hover-only disclosure for substantive wording.
- Where layout decisions require approximation for undated content, the UI should label that placement as undated / approximate-from-layout rather than presenting it as factual chronology.
