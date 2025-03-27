from galileo import galileo

def test_galileo_hello_world():
    g = g.dataset("my_compate")
    # mock
    g.mock(
        "get_stock_data",
        return_value=[
            {"date": "2020-01-01", "close": 100},
            {"date": "2020-01-02", "close": 200},
        ],
    )
    assert True