import json

"""
Code to compare two structured data objects.
"""
FNAME1 = "1_30s_4disc.json"
FNAME2 = "2_15s_2disc.json"

def compare_data(data1, data2):
    # Here I assume that these two data structures have the same
    # partition in the same order...
    for subset1, subset2 in zip(data1["model"], data2["model"]):
        assert subset1["sets"] == subset2["sets"]

        if subset1["sets"] is not None:
            for values1, values2 in zip(subset1["indices"], subset2["indices"]):
                for idx1, idx2 in zip(values1, values2):
                    assert abs(idx1 - idx2) < 1e-8
                    #assert idx1 == idx2


def main():
    with open(FNAME1, "r") as fp:
        data1 = json.load(fp)

    with open(FNAME2, "r") as fp:
        data2 = json.load(fp)

    compare_data(data1, data2)


if __name__ == "__main__":
    main()
