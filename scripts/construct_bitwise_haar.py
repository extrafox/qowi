
count = 0
for a in range(2):
    for b in range(2):
        for c in range(2):
            for d in range(2):
                ll = (a + b + c + d) % 2
                hl = ((a + b) % 2 - (c + d) % 2) % 2
                lh = ((a + c) % 2 - (b + d) % 2) % 2
                hh = ((a + d) % 2 - (b + c) % 2) % 2
                print(f"{count}: {a}{b}{c}{d} -> {ll}{hl}{lh}{hh}")
                count += 1

