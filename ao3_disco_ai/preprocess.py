import logging
import pickle

import typer

logging.basicConfig(level=logging.DEBUG)

app = typer.Typer()


@app.command()
def make_small_world():
    work_to_json, train_rows, test_rows = pickle.load(open("data/dataset.pkl", "rb"))
    logging.info("Found %d works.", len(work_to_json))
    logging.info("Found %d train rows.", len(train_rows))
    logging.info("Found %d test rows.", len(test_rows))

    train_rows = train_rows[:1000]
    test_rows = test_rows[:1000]
    valid_work_ids = set([row["work"] for row in train_rows + test_rows])
    for row in train_rows + test_rows:
        valid_work_ids.update(row["candidates"].keys())
    work_to_json = {k: v for k, v in work_to_json.items() if k in valid_work_ids}
    logging.info("Pruned to %d works.", len(work_to_json))
    logging.info("Pruned to %d train rows.", len(train_rows))
    logging.info("Pruned to %d test rows.", len(test_rows))

    with open("data/small-world.pkl", "wb") as fout:
        pickle.dump((work_to_json, train_rows, test_rows), fout)


if __name__ == "__main__":
    app()
