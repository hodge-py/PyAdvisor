import PyTrader

pyT = PyTrader.PyTrader()
"""
values = pyT.stockScreener(['A','AA','AACG','AACT','AADI','AAL','AAM','AAME','AAMI','AAOI','AAON','AAP',
                   'AAPG','AAPL','AAT','AB','ABAT','ABBV','ABCB','ABCL','ABEO','ABEV','ABG','ABL','ABLV','ABM','ABNB','ABOS','ABP','ABR','ABSI',
                   'ABT','ABTS','ABUS','ABVC','ABVE','ABVX','AC','ACA','ACAD','ACB','ACCD','ACCO','ACCS','ACDC','ACEL','ACET','ACGL','ACHC','ACHL','ACHR','ACHV',
                   'ACI','ACIC','ACIU','ACIW','ACLS','ACLX','ACM','ACMR','ACN','ACNB','ACNT','ACOG','ACON','ACR','ACRE','ACRS','ACRV','ACT','ACTG',
                   'ACTU','ACU','ACVA','ACXP','ADAG','ADAP','ADBE','ADC','ADCT','ADD','ADEA','ADGM','ADI','ADIL','ADM','ADMA'])
"""
pyT.generateValueReport(['MSFT','NVDA','AMZN','GOOGL','META'],save_as_csv=True)


pyT.generateTechnicalReport(['TER','MSFT','NVDA','AMZN','GOOGL','META','A','AA','AACG','AACT','AADI','AAL','AAM','AAME','AAMI','AAOI','AAON','AAP',
                   'AAPG','AAPL','AAT','AB','ABAT','ABBV','ABCB','ABCL','ABEO','ABEV','ABG','ABL','ABLV','ABM','ABNB','ABOS','ABP','ABR','ABSI',
                   'ABT','ABTS','ABUS','ABVC','ABVE','ABVX','AC','ACA','ACAD','ACB','ACCD','ACCO','ACCS','ACDC','ACEL','ACET','ACGL','ACHC','ACHL','ACHR','ACHV',
                   'ACI','ACIC','ACIU','ACIW','ACLS','ACLX','ACM','ACMR','ACN','ACNB','ACNT','ACOG','ACON','ACR','ACRE','ACRS','ACRV','ACT','ACTG',
                   'ACTU','ACU','ACVA','ACXP','ADAG','ADAP','ADBE','ADC','ADCT','ADD','ADEA','ADGM','ADI','ADIL','ADM','ADMA'])