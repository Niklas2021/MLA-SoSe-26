# Wohin Ergebnisse geschrieben werden. An einer Stelle, damit tune.py, autotune.py und
# check_coverage.py dieselbe Ablage benutzen -- der results/-Ordner wird nach dem Lauf
# vom Server zurueckkopiert (result_dgx/, result_3070/).
import csv
import os


def results_dir():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(d, exist_ok=True)
    return d


def write_csv(name, header, rows):
    # csv.writer statt join: Einsum-Strings ("cmk,ckn->cmn") und Notizen enthalten
    # Kommas, die muessen gequotet werden, sonst verrutschen die Spalten
    path = os.path.abspath(os.path.join(results_dir(), name))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(["" if v is None else v for v in r])
    print(f"[CSV: {path}]")
    return path


def write_log(lines, name):
    path = os.path.abspath(os.path.join(results_dir(), name))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[Log: {path}]")
    return path
