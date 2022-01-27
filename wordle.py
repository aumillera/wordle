import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from rich import print
from rich.progress import Progress


COLORS = dict(
    g="green",
    y="yellow",
    b="gray",
)
DEFAULT_DICT = "wordle_solutions.txt"
MAX_RECURSION = 0
progress = Progress(auto_refresh=False)


def prompt():
    print("[bold red]> [/bold red]", end="", flush=True)
    ret = input()
    return ret


def response_color_str(guess: str, response: str) -> str:
    colored_guess = str()
    for g, r in zip(guess, response):
        col = COLORS[r.lower()]
        colored_guess += f"[{col}]{g}[/{col}]"
    return colored_guess


def parse_response(guess: str, response: str) -> tuple:
    corr, incl, excl = set(), set(), set()
    guess = guess.lower()
    for i, c in enumerate(response.lower()):
        if c == "g":
            corr.add((i, guess[i]))
        if c == "y":
            incl.add((i, guess[i]))
        if c == "b":
            excl.add(guess[i])
    return corr, incl, excl


def filter_valid(words: np.ndarray, corr: set, incl: set, excl: set) -> np.ndarray:
    validity_list = list()
    for i, c in corr:
        valid = np.zeros(shape=words.shape, dtype=np.bool)
        valid[:, i] = words[:, i] == c
        validity_list.append(np.any(valid, axis=1))
    for i, c in incl:
        validity_list.append(np.any(words == c, axis=1))
        valid = np.zeros(shape=words.shape, dtype=np.bool)
        valid[:, i] = words[:, i] != c
        validity_list.append(np.any(valid, axis=1))
    for c in excl:
        validity_list.append(np.alltrue(words != c, axis=1))
    return np.alltrue(np.stack(validity_list), axis=0)


def generate_response(guess: str, target: str) -> str:
    response = str()
    for i, c in enumerate(guess):
        if target[i] == c:
            response += "g"
        elif c in target:
            response += "y"
        else:
            response += "b"

    assert len(response) == len(guess) == len(target)
    return response


def compute_letter_frequency(words: list):
    freq = defaultdict(float)
    for w in words:
        for c in set(w):
            freq[c] += 1
    for c in freq:
        freq[c] /= len(words)
    return freq


def compute_word_score(word: str, freq: dict):
    return sum([freq[c] for c in set(word)]) / float(len(word))


def recursive_compute_scores(
    words: np.ndarray,
    corr: set,
    incl: set,
    excl: set,
    freq: dict = None,
    last_guess: str = "",
    last_result: str = "",
    level: int = 0,
    max_level: int = 0
) -> np.ndarray:
    n_words = len(words)

    # End Condition
    if n_words <= 1:
        return np.array([float(n_words)])

    # Compute results for all possible solutions
    p_id = None
    if level == 0:
        color = "red"
        p_id = progress.add_task(".....", total=n_words)
    elif level == 1:
        color = "yellow"
    else:
        color = ""

    # Possible results pre-screening
    scores = np.ones(n_words)
    idx_to_check = range(n_words)
    if level != max_level and last_result == "":
        scores = recursive_compute_scores(words, corr, incl, excl, freq, last_guess, level=level, max_level=level)
        order = np.argsort(scores)[::-1]
        n_check = max(20, len(scores) // 20)
        idx_to_check = order[:n_check]
        scores[order[n_check:]] = 0.0

    if p_id is not None:
        progress.update(p_id, total=len(idx_to_check))
    for idx in idx_to_check:
        result = "".join(words[idx])
        if p_id is not None:
            progress.update(p_id, description=f"[bold {color}]R{level}: {result}[/bold {color}]", refresh=True)
        # progress.update(p_id, description=f"[bold {color}]{guess}[/bold {color}]")

        if last_result != "" and result != last_result:
            scores[idx] = 0.0
            continue

        sub_scores = np.ones(n_words) * (1.0 / n_words)

        # Sample through all possible results
        p_id2 = progress.add_task(".....", total=n_words)
        for jdx in range(n_words):
            # Skip if same word
            if idx == jdx:
                sub_scores[jdx] = 0.0
                continue

            guess = "".join(words[jdx])
            progress.update(p_id2, description=f"[bold {color}]G{level}: {guess}[/bold {color}]", refresh=True)

            # Do not pursue guess not different enough
            n_corr = max(len(set([t[1] for t in corr])), len(set([t[1] for t in incl])))
            if len(set(guess) - set(last_guess)) < min(2, len(set(result)) - n_corr):
                sub_scores[jdx] = 0.0
                continue

            new_corr, new_incl, new_excl = parse_response(guess, generate_response(guess, result))
            new_corr.update(corr), new_incl.update(incl), new_excl.update(excl)
            new_words = words[filter_valid(words, new_corr, new_incl, new_excl)]

            if len(new_words) < n_words and level < max_level and len(new_corr) > len(corr):
                sub_scores[jdx] = np.max(
                    recursive_compute_scores(
                        new_words,
                        new_corr,
                        new_incl,
                        new_excl,
                        freq=freq,
                        last_guess=guess,
                        last_result=result,
                        level=level + 1,
                        max_level=max_level
                    )
                )
            else:
                if level == max_level:
                    sub_scores[jdx] = 1.0 / len(new_words)
                else:
                    sub_scores[jdx] = 0.0

            if freq is not None:
                sub_scores[jdx] *= compute_word_score(guess, freq)

            progress.update(p_id2, advance=1, refresh=True)

        progress.remove_task(p_id2)

        best_guess_idx = np.argmax(sub_scores)
        scores[best_guess_idx] *= np.max(sub_scores)
        if freq is not None:
            for idx in range(len(scores)):
                scores[idx] *= compute_word_score(words[idx], freq)

        if p_id is not None:
            progress.update(p_id, advance=1, refresh=True)

    if p_id is not None:
        progress.remove_task(p_id)
    return scores


class Solver():
    """A word list solver.

    Takes inputs such as:

    - Guessed word
    - Correct letter & location
    - Correct letter
    - Wrong letter

    References
    ----------
    Based on the wordle game:
    https://www.powerlanguage.co.uk/wordle/
    """

    def __init__(self, words: list, max_tries: int, max_recursions: int):
        self._freq = compute_letter_frequency(words)
        self._ws = np.asarray([list(w) for w in words])
        self._scores = np.ones(self._ws.shape[0], dtype=np.int) * np.inf
        self._corr, self._incl, self._excl = set(), set(), set()
        self._max_tries = max_tries
        self._max_rec = max_recursions
        self._idx = 1

        print(f"\nInitialized solver with [bold red]{len(self._ws)}[/bold red] words.\n")

    def update(self, guess: str, response: str = None, mode: str = "s"):
        if response is None:
            print("Now what should I do with it without a proper response?")
            return

        colored_guess = response_color_str(guess, response)
        print(f"Got it, [bold]{colored_guess}[/bold].")

        # Parse Response
        corr, incl, excl = parse_response(guess, response)
        self._corr.update(corr)
        self._incl.update(incl)
        self._excl.update(excl)

        # Update wordlist
        n_old = len(self._ws)
        self._ws = self._ws[filter_valid(self._ws, corr, incl, excl)]
        print(f"Eliminated [bold green]{n_old - len(self._ws):d}[/bold green] possibitilies.")

        if self.solved():
            sol = "".join(self._ws[0])
            print(f"Solved after {self._idx} tries! The word is: [bold green]{sol}[/bold green]")
            return
        elif len(self._ws) == 0:
            print("[red]No words matching words were found, did you correctly type all responses?[/red]")
            exit(0)

        if mode == "s":
            with progress:
                self._scores = recursive_compute_scores(
                    self._ws,
                    self._corr,
                    self._incl,
                    self._excl,
                    freq=self._freq,
                    last_guess=guess,
                    max_level=self._max_rec
                )
        elif mode == "u":
            self._scores = np.asarray([compute_word_score(w, self._freq) for w in self._ws])
        else:
            print(f"Mode {mode} is unknown to me.")
            self._scores = np.asarray([compute_word_score(w, self._freq) for w in self._ws])

        self._scores *= 1.0 / (np.sum(self._scores) or 1.0)
        self.suggest()

        self._idx += 1

    def suggest(self):
        order = np.argsort(self._scores)[::-1]
        print(f"There are [bold red]{len(self._ws)}[/bold red] possible words left. Try one of those:\n")
        print(f"[bold]{'IDX':6s}{'SUGGESTION':12s}{'PROBABILITY':>12s}[/bold]")
        for i, (g, p) in enumerate(zip(self._ws[order][:10], self._scores[order][:10])):
            response = ["b"] * 5
            for i, c in self._corr:
                response[i] = "g"
            g_color = response_color_str(g, "".join(response))
            print(f"{str(i):6s}{''.join(g_color):12s}[yellow]{p * 100:11.1f}%[/yellow]")
        print("")


    def solved(self) -> bool:
        return len(self._ws) == 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Wordle Solver")
    parser.add_argument(
        "-d", "--dictionary", nargs="+", default=[Path(DEFAULT_DICT)], type=Path, required=False,
        help="Path to custom word list file. Words should be separated by newlines."
    )
    parser.add_argument(
        "-t", "--tries", default=6, type=int, required=False,
        help="Maximum number of tries."
    )
    parser.add_argument(
        "-r", "--recursions", default=0, type=int, required=False,
        help="Maxmimum number of recursions for score computation."
    )

    args = parser.parse_args()

    # Start new game
    words = list()
    for l in args.dictionary:
        words.extend(w.lower() for w in l.read_text().split("\n"))
    solver = Solver(words=words, max_tries=args.tries, max_recursions=args.recursions)

    while not solver.solved():
        random_response = "".join(np.random.choice(list(COLORS.keys())) for r in range(5))
        random_word = np.random.choice(words)
        colored_guess = response_color_str(random_word, random_response)
        print(f"Enter your guess and the response (and the mode), all separated by a space, e.g.: [bold]{colored_guess} {random_response}[/bold]")
        print("Mode: [bold]s[/bold] suggest a new word (default) or [bold]u[/bold] update wordlist (fast).")
        print("[green]:green_square: [bold]g[/bold] Correct letter & location[/green]")
        print("[yellow]:yellow_square: [bold]y[/bold] Correct letter[/yellow]")
        print(":black_large_square: [bold]b[/bold] Incorrect letter\n")
        response = prompt().split(" ")
        solver.update(*response)

    exit(0)