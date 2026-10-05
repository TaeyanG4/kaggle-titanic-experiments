"""Build a human-readable peer-evidence report for v10 vs trusted conflicts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
V13_DIR = BASE_DIR / "exports" / "v13"


def surname(name: str) -> str:
    s = str(name)
    if "(" in s:
        s = s.split("(")[0]
    return re.sub(r"[^\w\s-]", "", s.split(",")[0]).strip()


def peer_text(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    parts = []
    for _, r in df.iterrows():
        parts.append(
            f"{int(r['PassengerId'])}:{r['Name']}[S={int(r['Survived'])}]"
        )
    return " | ".join(parts)


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    conflict = pd.read_csv(V13_DIR / "test_v10_v5_conflicts.csv")

    train["FamilyKey"] = train["Name"].map(surname)
    test["FamilyKey"] = test["Name"].map(surname)

    rows = []
    for _, c in conflict.iterrows():
        pid = int(c["PassengerId"])
        trow = test.loc[test["PassengerId"] == pid].iloc[0]
        fam = trow["FamilyKey"]
        ticket = str(trow["Ticket"])
        family_peers = train[train["FamilyKey"] == fam].copy()
        ticket_peers = train[train["Ticket"].astype(str) == ticket].copy()
        rows.append(
            {
                "PassengerId": pid,
                "Name": trow["Name"],
                "Sex": trow["Sex"],
                "Age": trow["Age"],
                "Pclass": int(trow["Pclass"]),
                "Family": fam,
                "Ticket": ticket,
                "v10": int(c["v10"]),
                "v5": int(c["v5"]),
                "TabPFN3_FF": int(c["TabPFN3_FF"]),
                "FamilyAvailable": float(c["FamilyAvailable"]),
                "FamilyRate": float(c["FamilyRate"]),
                "TicketAvailable": float(c["TicketAvailable"]),
                "TicketRate": float(c["TicketRate"]),
                "SurvivalRate": float(c["SurvivalRate"]),
                "SurvivalRateNA": float(c["SurvivalRateNA"]),
                "train_family_peer_count": len(family_peers),
                "train_family_survivors": int(family_peers["Survived"].sum()),
                "train_ticket_peer_count": len(ticket_peers),
                "train_ticket_survivors": int(ticket_peers["Survived"].sum()),
                "family_peers": peer_text(family_peers),
                "ticket_peers": peer_text(ticket_peers),
            }
        )

    detail = pd.DataFrame(rows)
    detail.to_csv(V13_DIR / "test_conflict_peer_evidence.csv", index=False)

    summary = (
        detail.groupby(["SurvivalRateNA", "Sex"], dropna=False)
        .agg(
            n=("PassengerId", "size"),
            v10_alive=("v10", "sum"),
            v5_alive=("v5", "sum"),
            tabpfn_alive=("TabPFN3_FF", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(V13_DIR / "test_conflict_group_summary.csv", index=False)

    no_group = detail[detail["SurvivalRateNA"] == 0]
    full_group = detail[detail["SurvivalRateNA"] == 1]
    partial_group = detail[
        (detail["SurvivalRateNA"] > 0) & (detail["SurvivalRateNA"] < 1)
    ]

    report = [
        "# v10 vs trusted conflict audit",
        "",
        "## What the v10 test-aware signal is",
        "",
        "v10 only creates Family/Ticket survival-rate mappings for groups that",
        "occur in both training data and the current inference test set. The",
        "survival value itself comes only from train labels. SurvivalRateNA is:",
        "",
        "- 0.0: neither family nor ticket supplies a train-label peer signal",
        "- 0.5: exactly one of family/ticket supplies a peer signal",
        "- 1.0: both family and ticket supply peer signals",
        "",
        "This makes the preprocessing transductive: it uses the identity of the",
        "actual test groups, but not their labels.",
        "",
        "## Actual v10 vs v5 test conflicts",
        "",
        f"- total: {len(detail)}",
        f"- no group signal (NA=0): {len(no_group)}",
        f"- partial group signal (NA=0.5): {len(partial_group)}",
        f"- both family and ticket signals (NA=1): {len(full_group)}",
        "",
        "## Conservative guard candidate",
        "",
        "The lowest-degree candidate keeps v10 everywhere except when:",
        "",
        "1. v10 and trusted v5 disagree;",
        "2. SurvivalRateNA == 0 (no Family/Ticket evidence);",
        "3. v5 and TabPFN v3 FamilyFare agree on the opposite label.",
        "",
        "On the real test this changes only PassengerId 929 and 1231.",
        "It is saved as submission_v13_guard_v5_tab_no_group.csv and has not",
        "been submitted.",
        "",
        "## Important validation caveat",
        "",
        "A fold-safe transductive analogue scored below v5 locally. Therefore",
        "this selector is a Public-LB hypothesis, not a promoted trusted model.",
        "Do not replace the v5 trusted champion without external evidence.",
    ]
    (V13_DIR / "conflict_audit_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print("=== conflict group summary ===")
    print(summary.to_string(index=False))
    print("\n=== strongest peer-evidence conflicts ===")
    cols = [
        "PassengerId",
        "Name",
        "v10",
        "v5",
        "TabPFN3_FF",
        "SurvivalRate",
        "SurvivalRateNA",
        "train_family_peer_count",
        "train_family_survivors",
        "train_ticket_peer_count",
        "train_ticket_survivors",
    ]
    print(
        detail.sort_values(
            ["SurvivalRateNA", "PassengerId"], ascending=[False, True]
        )[cols].to_string(index=False)
    )


if __name__ == "__main__":
    main()
