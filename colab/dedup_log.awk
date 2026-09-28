# Skraca wyjście komórki Colaba: linie różniące się tylko liczbami pokazuje najwyżej 3 razy.
# Komunikaty skyhunt z nazwą pliku ("INFO [plik] …") i błędy przechodzą zawsze.
# Pełny, nieskrócony log zapisuje `tee` (skyhunt_run.log na Drive).
{
    key = $0
    sub(/^[0-9][0-9]:[0-9][0-9]:[0-9][0-9] /, "", key)
    if (key ~ /^(INFO|WARNING|ERROR) \[/ || key ~ /^ERROR/) { print; fflush(); next }
    gsub(/[0-9]+/, "#", key)
    n[key]++
    if (n[key] <= 3) print
    else if (n[key] == 4) print "    … kolejne powtórzenia tej linii tylko w skyhunt_run.log"
    fflush()
}
