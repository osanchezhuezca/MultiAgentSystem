"""Seed SQLite from the Kaggle customer support tickets CSV."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.config import settings
from .models import Base, Customer, SupportTicket


def _clean_str(value):
    """Return None for NaN, otherwise a stripped string."""
    if pd.isna(value):
        return None
    return str(value).strip()


def _parse_datetime(value):
    """Best-effort parse of 'Date of Purchase' — returns None on failure."""
    if pd.isna(value):
        return None
    try:
        return pd.to_datetime(value, errors="coerce")
    except Exception:
        return None


def _parse_rating(value):
    """Satisfaction ratings are 1-5; return None if missing or non-numeric."""
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def seed_database(
    csv_path: str | None = None,
    db_url: str | None = None,
) -> None:
    csv_path = csv_path or settings.tickets_csv
    db_url = db_url or settings.database_url

    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_file}")

    print(f"Reading {csv_file} ...")
    df = pd.read_csv(csv_file)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    # --- Drop rows missing the fields we truly need ---
    required = ["Customer Email", "Customer Name", "Ticket Subject"]
    before = len(df)
    df = df.dropna(subset=required)
    if len(df) < before:
        print(f"  Dropped {before - len(df)} rows missing required fields.")

    engine = create_engine(db_url)
    Base.metadata.drop_all(engine)   # fresh rebuild on every seed
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        # ---------- Build customers (dedupe on email) ----------
        customers_df = (
            df[["Customer Email", "Customer Name", "Customer Age", "Customer Gender"]]
            .drop_duplicates(subset=["Customer Email"])
            .reset_index(drop=True)
        )

        email_to_id: dict[str, int] = {}
        for _, row in customers_df.iterrows():
            email = _clean_str(row["Customer Email"])
            if not email:
                continue
            cust = Customer(
                name=_clean_str(row["Customer Name"]) or "Unknown",
                email=email,
                age=int(row["Customer Age"]) if pd.notna(row["Customer Age"]) else None,
                gender=_clean_str(row["Customer Gender"]),
            )
            session.add(cust)
            session.flush()                 # get auto-generated id
            email_to_id[email] = cust.id

        print(f"  Inserted {len(email_to_id)} customers.")

        # ---------- Build tickets ----------
        ticket_count = 0
        for _, row in df.iterrows():
            email = _clean_str(row["Customer Email"])
            cust_id = email_to_id.get(email)
            if cust_id is None:
                continue

            ticket = SupportTicket(
                customer_id=cust_id,
                product_purchased=_clean_str(row.get("Product Purchased")),
                date_of_purchase=_parse_datetime(row.get("Date of Purchase")),
                ticket_type=_clean_str(row.get("Ticket Type")),
                subject=_clean_str(row.get("Ticket Subject")),
                description=_clean_str(row.get("Ticket Description")),
                status=_clean_str(row.get("Ticket Status")),
                resolution=_clean_str(row.get("Resolution")),
                priority=_clean_str(row.get("Ticket Priority")),
                channel=_clean_str(row.get("Ticket Channel")),
                first_response_time=_clean_str(row.get("First Response Time")),
                time_to_resolution=_clean_str(row.get("Time to Resolution")),
                satisfaction_rating=_parse_rating(row.get("Customer Satisfaction Rating")),
            )
            session.add(ticket)
            ticket_count += 1

            if ticket_count % 2000 == 0:
                session.flush()
                print(f"  ... {ticket_count} tickets staged")

        session.commit()

    print(f"Done. {len(email_to_id)} customers, {ticket_count} tickets.")


if __name__ == "__main__":
    seed_database()