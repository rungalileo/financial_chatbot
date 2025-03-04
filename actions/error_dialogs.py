ADDENDUM = f"""
You can also ask me things like stuff like:
<ul>
    <li>Give me information on Meta</li>
    <li>How did the market do in 2024</li>
    <li>How can I start investing in the stock market?</li>
    <li>What are the best stocks to invest in right now?</li>
    <li>Any general questions about the stock market</li>
</ul>
"""

CAN_ONLY_DO_PERF_INFO_ON_SPECIFIC_STOCKS = f"""
    <span style='color: red;'>I can only do performance information on specific stocks. Please provide a company name or a stock symbol!</span>
    {ADDENDUM}
"""

PLEASE_SPECIFY_A_STOCK = f"""
    <span style='color: red;'>Please specify a stock symbol or company name.</span>
    {ADDENDUM}
"""
