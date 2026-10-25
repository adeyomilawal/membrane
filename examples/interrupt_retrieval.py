"""Parse model-emitted interrupt tags."""

from membrane import contains_interrupt, parse_interrupt, strip_all_interrupts


def main() -> None:
    text = "I need more context. <SYSTEM_INTERRUPT_FETCH[User's timezone?]> Continuing."
    if contains_interrupt(text):
        parsed = parse_interrupt(text)
        if parsed:
            before, query, after = parsed
            print("query:", query)
            print("strip:", strip_all_interrupts(text))


if __name__ == "__main__":
    main()
