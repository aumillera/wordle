import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from rich import print
from rich.progress import track, Progress, TaskID


DEFAULT_DICT = "wordle_all_words.txt"
MAX_RECURSION = 0
progress = Progress(auto_refresh=False)


def prompt():
    print("[bold red]> [/bold red]", end="", flush=True)
    ret = input()
    return ret


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
        validity_list.append((np.char.find(words[:, i], c) + 1).astype(np.bool))
    for i, c in incl:
        validity_list.append(np.invert((np.char.find(words[:, i], c) + 1).astype(np.bool)))
        validity_list.append(np.sum(np.char.find(words, c) + 1, axis=1, dtype=np.bool))
    for c in excl:
        validity_list.append(np.invert(np.sum(np.char.find(words, c) + 1, axis=1, dtype=np.bool)))
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


def recursive_compute_scores(words: np.ndarray, corr: set, incl: set, excl: set, level: int = 0, max_level: int = 0) -> np.ndarray:
    n_words = len(words)

    # End Condition
    if n_words <= 1:
        return np.array([float(n_words)])

    # Compute results for all possible solutions
    scores = np.ones(n_words)
    color = "red" if level == 0 else "yellow" if level == 1 else ""
    p_id = progress.add_task("Thinking", total=n_words)
    for idx in range(n_words):
        guess = "".join(words[idx])
        progress.update(p_id, description=f"[bold {color}]{guess}[/bold {color}]")
        sub_scores = np.ones(n_words) * (1.0 / n_words)

        # Sample through all possible results
        for jdx in range(n_words):
            # Skip if same word
            if idx == jdx:
                sub_scores[idx] = 0.0
                continue

            result = "".join(words[jdx])
            new_corr, new_incl, new_excl = parse_response(guess, generate_response(guess, result))
            new_corr.update(corr), new_incl.update(incl), new_excl.update(excl)
            new_words = words[filter_valid(words, new_corr, new_incl, new_excl)]

            if len(new_words) < n_words // 2 and level < max_level:
                sub_scores[idx] *= np.max(recursive_compute_scores(new_words, new_corr, new_incl, new_excl, level + 1, max_level))
            else:
                sub_scores[idx] = 1.0 / len(new_words)

        scores[idx] *= np.max(sub_scores)
        progress.update(p_id, advance=1, refresh=True)

    progress.remove_task(p_id)
    return scores


def compute_scores(words: np.ndarray, corr: set, incl: set, excl: set) -> np.ndarray:
    n_words = len(words)

    # End Condition
    if n_words <= 1:
        return np.array([float(n_words)])

    # Compute results for all possible solutions
    scores = np.ones(n_words)
    p_id = progress.add_task("Thinking", total=n_words)
    for idx in range(n_words):
        guess = "".join(words[idx])
        progress.update(p_id, description=f"[bold red]{guess}[/bold red]")
        sub_scores = np.ones(n_words) * (1.0 / n_words)

        # Sample through all possible results
        for jdx in range(n_words):
            # Skip if same word
            if idx == jdx:
                sub_scores[idx] = 0.0
                continue

            result = "".join(words[jdx])
            new_corr, new_incl, new_excl = parse_response(guess, generate_response(guess, result))
            new_corr.update(corr), new_incl.update(incl), new_excl.update(excl)
            new_words = words[filter_valid(words, new_corr, new_incl, new_excl)]
            sub_scores[idx] = 1.0 / len(new_words)

        scores[idx] *= np.max(sub_scores)
        progress.update(p_id, advance=1, refresh=True)

    progress.remove_task(p_id)
    return scores


def compute_letter_frequency(words: list) -> dict:
    freq = defaultdict(float)
    for w in words:
        for l in set(w):
            freq[l] += 1
    for l in freq:
        freq[l] /= len(words)

    return freq


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

    def __init__(self, word_list: Path, max_tries: int, max_recursions: int):
        words = [w.lower() for w in word_list.read_text().split("\n")]
        self._ws = np.asarray([list(w) for w in words])
        self._scores = np.ones(self._ws.shape[0], dtype=np.int) * np.inf
        self._corr, self._incl, self._excl = set(), set(), set()
        self._max_tries = max_tries
        self._max_rec = max_recursions
        self._idx = 0
        self._freq = compute_letter_frequency(words)

        print(f"\nInitialized solver with [bold red]{len(self._ws)}[/bold red] words.\n")

    def update(self, guess: str, response: str):
        # Parse Response
        corr, incl, excl = parse_response(guess, response)
        self._corr.update(corr)
        self._incl.update(incl)
        self._excl.update(excl)

        # Update wordlist
        self._ws = self._ws[filter_valid(self._ws, corr, incl, excl)]

        if self.solved():
            sol = "".join(self._ws[0])
            print(f"Solved after {self._idx} tries! The word is: [bold green]{sol}[/bold green]")
            return

        with progress:
            self._scores = recursive_compute_scores(self._ws, self._corr, self._incl, self._excl, max_level=self._max_rec)
        self._scores *= 1.0 / np.sum(self._scores)
        self.suggest()

        self._idx += 1

    def suggest(self):
        order = np.argsort(self._scores)[::-1]
        print(f"There are [bold red]{len(self._ws)}[/bold red] possible words left. Try one of those:\n")
        print(f"[bold]{'IDX':6s}{'SUGGESTION':12s}{'PROBABILITY':>12s}[/bold]")
        for i, (g, p) in enumerate(zip(self._ws[order][:10], self._scores[order][:10])):
            print(f"{str(i):6s}{''.join(g):12s}{p * 100:11.1f}%")


    def solved(self) -> bool:
        return len(self._ws) == 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Wordle Solver")
    parser.add_argument(
        "-d", "--dictionary", default=DEFAULT_DICT, type=Path, required=False,
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
    solver = Solver(word_list=args.dictionary, max_tries=args.tries, max_recursions=args.recursions)

    while not solver.solved():
        print("Enter your guess and the response separated by a space, e.g.: [bold][green]g[/green]ue[yellow]ss[/yellow] gbbyy[/bold]")
        print("[green]:green_square: [bold]g[/bold]: Correct letter & location[/green]")
        print("[yellow]:yellow_square: [bold]y[/bold] Correct letter[/yellow]")
        print(":black_large_square: [bold]b[/bold] Incorrect letter\n")
        guess, response = prompt().split(" ")
        solver.update(guess, response)

    exit(0)