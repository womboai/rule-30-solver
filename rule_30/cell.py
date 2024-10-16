def next_row(row: int):
    return row ^ ((row << 1) | (row << 2))
