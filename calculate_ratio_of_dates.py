from datetime import datetime


def calculate_ratio(start_dt, split_dt, end_dt):

    total_days = (end_dt - start_dt).days
    back_test_days = (end_dt - split_dt).days

    ratio = 1.0 - (back_test_days / total_days)

    print(start_dt)
    print(split_dt)
    print(end_dt)
    print(ratio)


if __name__ == "__main__":

    # # Jiang 1
    start_dt = datetime.strptime("20150220", "%Y%m%d")
    split_dt = datetime.strptime("20160907", "%Y%m%d")
    end_dt = datetime.strptime("20161028", "%Y%m%d")

    # Jiang 2
    # start_dt = datetime.strptime("20150220", "%Y%m%d")
    # split_dt = datetime.strptime("20161208", "%Y%m%d")
    # end_dt = datetime.strptime("20170128", "%Y%m%d")

    # # Jiang 3
    # start_dt = datetime.strptime("20150501", "%Y%m%d")
    # split_dt = datetime.strptime("20170307", "%Y%m%d")
    # end_dt = datetime.strptime("20170427", "%Y%m%d")

    calculate_ratio(start_dt, split_dt, end_dt)
