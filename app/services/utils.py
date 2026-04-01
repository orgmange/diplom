from typing import Optional


def detect_doc_type(filename: str) -> Optional[str]:
    """Определяет тип документа по имени файла."""
    lowered = filename.lower()
    rules = (
        ("dogovor_kupli_kv", "dogovor_kupli_prodazhi_kv"),
        ("dogovor_kupli", "dogovor_prodagi_machini"),
        ("passport", "passport"),
        ("паспорт", "passport"),
        ("prava", "driver_license"),
        ("права", "driver_license"),
        ("driver", "driver_license"),
        ("snils", "snils"),
        ("снилс", "snils"),
        ("svid", "birth_certificate"),
        ("свид", "birth_certificate"),
        ("birth", "birth_certificate"),
        ("diplom", "diplom_bakalavra"),
        ("renal", "dogovor_arendi_kv"),
        ("inn", "inn"),
        ("kvit", "kvitancia"),
        ("zagran", "zagran_passport"),
    )
    for token, doc_type in rules:
        if token in lowered:
            return doc_type
    return None
