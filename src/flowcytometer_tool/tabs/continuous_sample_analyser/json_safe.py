import json
import datetime as _datetime
from pathlib import Path

import numpy as np
import pandas as pd


def json_safe(
    obj,
    *,
    debug=False,
    path="root",
    max_repr=300,
    max_items_preview=10,
):
    """
    Recursively convert pandas/numpy/path/datetime objects for json.dump/json.dumps.

    If debug=True, prints every object visited:
      - path within the packet
      - Python type
      - repr preview
      - container size/shape where available
      - conversion action taken

    Example:
        packet = json_safe(packet, debug=True)
        json.dump(packet, f, indent=2)
    """

    def _preview(value):
        try:
            r = repr(value)
        except Exception as e:
            r = f"<repr failed: {type(e).__name__}: {e}>"

        if len(r) > max_repr:
            return r[:max_repr] + " ... <truncated>"
        return r

    def _print(action, value=None, extra=None):
        if not debug:
            return

        value_type = type(value).__name__ if value is not None else type(obj).__name__
        mod = type(value).__module__ if value is not None else type(obj).__module__
        full_type = f"{mod}.{value_type}"

        msg = f"[json_safe] path={path} | type={full_type} | action={action}"

        if extra:
            msg += f" | {extra}"

        try:
            msg += f" | repr={_preview(obj)}"
        except Exception:
            pass

        print(msg)

    # ------------------------------------------------------------------
    # Root entry print for every object
    # ------------------------------------------------------------------
    if debug:
        extra_bits = []

        try:
            if hasattr(obj, "shape"):
                extra_bits.append(f"shape={getattr(obj, 'shape')}")
        except Exception:
            pass

        try:
            if hasattr(obj, "dtype"):
                extra_bits.append(f"dtype={getattr(obj, 'dtype')}")
        except Exception:
            pass

        try:
            if hasattr(obj, "dtypes"):
                extra_bits.append(f"dtypes={getattr(obj, 'dtypes')}")
        except Exception:
            pass

        try:
            if hasattr(obj, "__len__") and not isinstance(obj, (str, bytes, bytearray)):
                extra_bits.append(f"len={len(obj)}")
        except Exception:
            pass

        _print("VISIT", obj, extra="; ".join(extra_bits) if extra_bits else None)

    # ------------------------------------------------------------------
    # Already JSON-safe primitives
    # ------------------------------------------------------------------
    if obj is None:
        _print("None -> None", obj)
        return None

    if isinstance(obj, bool):
        _print("bool -> bool", obj)
        return obj

    if isinstance(obj, str):
        _print("str -> str", obj)
        return obj

    if isinstance(obj, int) and not isinstance(obj, bool):
        _print("int -> int", obj)
        return obj

    if isinstance(obj, float):
        if np.isfinite(obj):
            _print("finite float -> float", obj)
            return obj
        _print("non-finite float -> None", obj)
        return None

    # ------------------------------------------------------------------
    # pandas scalar missing / datetime / timedelta
    # ------------------------------------------------------------------
    if obj is pd.NaT:
        _print("pd.NaT -> None", obj)
        return None

    if isinstance(obj, pd.Timestamp):
        if pd.isna(obj):
            _print("pd.Timestamp NaT-like -> None", obj)
            return None
        value = obj.isoformat()
        _print("pd.Timestamp -> isoformat string", obj, extra=f"converted={value!r}")
        return value

    if isinstance(obj, pd.Timedelta):
        if pd.isna(obj):
            _print("pd.Timedelta NaT-like -> None", obj)
            return None
        value = obj.isoformat()
        _print("pd.Timedelta -> isoformat string", obj, extra=f"converted={value!r}")
        return value

    # ------------------------------------------------------------------
    # Python datetime-like types
    # ------------------------------------------------------------------
    if isinstance(obj, _datetime.datetime):
        value = obj.isoformat()
        _print("datetime.datetime -> isoformat string", obj, extra=f"converted={value!r}")
        return value

    if isinstance(obj, _datetime.date):
        value = obj.isoformat()
        _print("datetime.date -> isoformat string", obj, extra=f"converted={value!r}")
        return value

    if isinstance(obj, _datetime.time):
        value = obj.isoformat()
        _print("datetime.time -> isoformat string", obj, extra=f"converted={value!r}")
        return value

    if isinstance(obj, _datetime.timedelta):
        value = obj.total_seconds()
        _print("datetime.timedelta -> total seconds", obj, extra=f"converted={value!r}")
        return value

    # ------------------------------------------------------------------
    # pathlib
    # ------------------------------------------------------------------
    if isinstance(obj, Path):
        value = str(obj)
        _print("Path -> str", obj, extra=f"converted={value!r}")
        return value

    # ------------------------------------------------------------------
    # numpy scalar datetime/timedelta
    # ------------------------------------------------------------------
    if isinstance(obj, np.datetime64):
        try:
            if np.isnat(obj):
                _print("np.datetime64 NaT -> None", obj)
                return None
        except Exception as e:
            _print("np.datetime64 np.isnat check failed", obj, extra=f"{type(e).__name__}: {e}")

        try:
            value = pd.Timestamp(obj).isoformat()
            _print("np.datetime64 -> pd.Timestamp.isoformat string", obj, extra=f"converted={value!r}")
            return value
        except Exception as e:
            value = str(obj)
            _print("np.datetime64 fallback -> str", obj, extra=f"{type(e).__name__}: {e}; converted={value!r}")
            return value

    if isinstance(obj, np.timedelta64):
        try:
            if np.isnat(obj):
                _print("np.timedelta64 NaT -> None", obj)
                return None
        except Exception as e:
            _print("np.timedelta64 np.isnat check failed", obj, extra=f"{type(e).__name__}: {e}")

        try:
            value = pd.Timedelta(obj).isoformat()
            _print("np.timedelta64 -> pd.Timedelta.isoformat string", obj, extra=f"converted={value!r}")
            return value
        except Exception as e:
            value = str(obj)
            _print("np.timedelta64 fallback -> str", obj, extra=f"{type(e).__name__}: {e}; converted={value!r}")
            return value

    # ------------------------------------------------------------------
    # numpy scalar generic types
    # ------------------------------------------------------------------
    if isinstance(obj, np.generic):
        try:
            value = obj.item()
            _print(
                "np.generic -> item(), then recurse",
                obj,
                extra=f"item_type={type(value).__module__}.{type(value).__name__}; item_repr={_preview(value)}",
            )
            return json_safe(
                value,
                debug=debug,
                path=path,
                max_repr=max_repr,
                max_items_preview=max_items_preview,
            )
        except Exception as e:
            value = str(obj)
            _print("np.generic fallback -> str", obj, extra=f"{type(e).__name__}: {e}; converted={value!r}")
            return value

    # ------------------------------------------------------------------
    # pandas containers
    # ------------------------------------------------------------------
    if isinstance(obj, pd.DataFrame):
        _print(
            "DataFrame -> list[dict] records, then recurse",
            obj,
            extra=f"shape={obj.shape}; columns={list(obj.columns)[:max_items_preview]!r}",
        )
        records = obj.to_dict(orient="records")
        return json_safe(
            records,
            debug=debug,
            path=f"{path}.<DataFrameRecords>",
            max_repr=max_repr,
            max_items_preview=max_items_preview,
        )

    if isinstance(obj, pd.Series):
        _print(
            "Series -> dict, then recurse",
            obj,
            extra=f"len={len(obj)}; name={obj.name!r}; dtype={obj.dtype}",
        )
        as_dict = obj.to_dict()
        return json_safe(
            as_dict,
            debug=debug,
            path=f"{path}.<SeriesDict>",
            max_repr=max_repr,
            max_items_preview=max_items_preview,
        )

    if isinstance(obj, pd.Index):
        _print(
            "Index -> list, then recurse",
            obj,
            extra=f"len={len(obj)}; dtype={obj.dtype}",
        )
        return json_safe(
            obj.tolist(),
            debug=debug,
            path=f"{path}.<IndexList>",
            max_repr=max_repr,
            max_items_preview=max_items_preview,
        )

    # ------------------------------------------------------------------
    # numpy arrays
    # ------------------------------------------------------------------
    if isinstance(obj, np.ndarray):
        _print(
            "ndarray -> list, then recurse",
            obj,
            extra=f"shape={obj.shape}; dtype={obj.dtype}",
        )
        return json_safe(
            obj.tolist(),
            debug=debug,
            path=f"{path}.<ndarrayList>",
            max_repr=max_repr,
            max_items_preview=max_items_preview,
        )

    # ------------------------------------------------------------------
    # dicts
    # ------------------------------------------------------------------
    if isinstance(obj, dict):
        _print("dict -> recurse into keys and values", obj, extra=f"len={len(obj)}")

        safe = {}

        for i, (k, v) in enumerate(obj.items()):
            key_path = f"{path}.<key:{i}>"
            value_key_preview = _preview(k)

            if debug:
                print(
                    f"[json_safe] path={path} | dict_item={i} | "
                    f"raw_key_type={type(k).__module__}.{type(k).__name__} | "
                    f"raw_key_repr={value_key_preview}"
                )

            safe_key_obj = json_safe(
                k,
                debug=debug,
                path=key_path,
                max_repr=max_repr,
                max_items_preview=max_items_preview,
            )

            # JSON keys need to be strings in practice. Force this so weird
            # Timestamp keys, tuple keys, numpy keys, etc. cannot leak through.
            safe_key = str(safe_key_obj)

            if debug:
                print(
                    f"[json_safe] path={path} | dict_item={i} | "
                    f"safe_key={safe_key!r}"
                )

            safe[safe_key] = json_safe(
                v,
                debug=debug,
                path=f"{path}.{safe_key}",
                max_repr=max_repr,
                max_items_preview=max_items_preview,
            )

        return safe

    # ------------------------------------------------------------------
    # list / tuple / set
    # ------------------------------------------------------------------
    if isinstance(obj, (list, tuple, set)):
        container_type = type(obj).__name__
        _print(f"{container_type} -> list, recurse into items", obj, extra=f"len={len(obj)}")

        return [
            json_safe(
                v,
                debug=debug,
                path=f"{path}[{i}]",
                max_repr=max_repr,
                max_items_preview=max_items_preview,
            )
            for i, v in enumerate(obj)
        ]

    # ------------------------------------------------------------------
    # Generic pandas missing values
    # Keep this late because pd.isna(container) can return arrays.
    # ------------------------------------------------------------------
    try:
        missing = pd.isna(obj)
        if isinstance(missing, (bool, np.bool_)) and missing:
            _print("pd.isna(obj) is True -> None", obj)
            return None
        elif debug:
            print(
                f"[json_safe] path={path} | pd.isna check result type="
                f"{type(missing).__module__}.{type(missing).__name__} | result_repr={_preview(missing)}"
            )
    except Exception as e:
        if debug:
            print(
                f"[json_safe] path={path} | pd.isna check raised "
                f"{type(e).__name__}: {e}"
            )