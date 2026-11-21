import h5py
import numpy as np
import pandas as pd

from bmtool.util.util import load_nodes


def _write_net(tmp_path):
    """Two node groups; only group 0 defines the bool flag ``tone``, both define ``x``."""
    h5 = tmp_path / "net_nodes.h5"
    csv = tmp_path / "net_node_types.csv"
    pd.DataFrame({"node_type_id": [100, 101], "pop_name": ["PN", "PV"]}).to_csv(
        csv, sep=" ", index=False
    )
    with h5py.File(h5, "w") as f:
        g = f.create_group("nodes/net")
        g["node_id"] = np.arange(5)
        g["node_type_id"] = np.array([100, 100, 100, 101, 101])
        g["node_group_id"] = np.array([0, 0, 0, 1, 1])
        g["node_group_index"] = np.array([0, 1, 2, 0, 1])
        g["0/tone"] = np.array([True, False, True])
        g["0/x"] = np.array([1.0, 2.0, 3.0])
        g["1/x"] = np.array([4.0, 5.0])
    return str(h5), str(csv)


def test_bool_flag_missing_in_group_is_false(tmp_path):
    pop, df = load_nodes(*_write_net(tmp_path))
    assert pop == "net"
    assert df["tone"].dtype == bool
    assert df["tone"].tolist() == [True, False, True, False, False]


def test_numeric_property_from_all_groups(tmp_path):
    _, df = load_nodes(*_write_net(tmp_path))
    np.testing.assert_allclose(df["x"].values, [1, 2, 3, 4, 5])
    assert df["pop_name"].tolist() == ["PN"] * 3 + ["PV"] * 2
