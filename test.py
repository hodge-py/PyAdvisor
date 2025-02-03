import PyTrader

pyT = PyTrader.PyTrader()
"""
values = pyT.stockScreener(['A','AA','AACG','AACT','AADI','AAL','AAM','AAME','AAMI','AAOI','AAON','AAP',
                   'AAPG','AAPL','AAT','AB','ABAT','ABBV','ABCB','ABCL','ABEO','ABEV','ABG','ABL','ABLV','ABM','ABNB','ABOS','ABP','ABR','ABSI',
                   'ABT','ABTS','ABUS','ABVC','ABVE','ABVX','AC','ACA','ACAD','ACB','ACCD','ACCO','ACCS','ACDC','ACEL','ACET','ACGL','ACHC','ACHL','ACHR','ACHV',
                   'ACI','ACIC','ACIU','ACIW','ACLS','ACLX','ACM','ACMR','ACN','ACNB','ACNT','ACOG','ACON','ACR','ACRE','ACRS','ACRV','ACT','ACTG',
                   'ACTU','ACU','ACVA','ACXP','ADAG','ADAP','ADBE','ADC','ADCT','ADD','ADEA','ADGM','ADI','ADIL','ADM','ADMA'])
"""
pyT.generateValueReport(['GE','CAT','RTX','UNP','HON','BA','DE','ETN','LMT','UPS','RELX','PH','WM','MMM','TT','CTAS','ITW','TRI',
                         'TDG','CP','EMR','NOC','GD','RSG','CNI','FDX','CSX','CARR','PCAR','NSC','CPRT','GWW','JCI','HWM','AXON','URI','CMI','WCN','PWR','VRT','DAL','AME','FAST','VRSK','LHX','ODFL','OTIS','IR','FERG','WAB','UAL','EFX','HEI','FER','ROK','XYL','HEI.A','GPN','DOV','VLTO','RYAAY','HUBB','LII','EME','BLDR','SNA','LUV','WSO.B','WSO','AER','CSL','J','SYM','JBHT','PNR','MAS','GFL','IEX','RBA','BAH','CNH','EXPD','OC','XPO','FIX','ZTO','RKLB','UHAL','GGG','TXT','ACM','SWK','ESLT','CW','POOL','UHAL.B','SAIA','NDSN','CLH','RTO','ITT','CHRW','ALLE','MTZ','LECO','CNM','TFII','AAL','WWD','RBC','ULS','NVT','APG','RRX','BWXT','FTAI','ARMK','AYI','BLD','AIT','TTEK','AOS','CR','CRS','AAON','WMS','PAC','DRS','KNX','MIDD','LTM','WCC','ALK','SARO','MLI','FBIN','GTLS','GNRC','STN','DCI','TTC','FLR','FLS','LPX','ASR','ATI','DLB','TREX','AGCO','HII','OSK','CAE','ERJ','ESAB','LOAR','AZEK','BECN','KBR','FCN','ADT','WTS','SPXC','WSC','JBTM','CWST','R','ZWS','VMI','AWI','RHI','MSA','SITE','JOBY','KEX','SMR','FSS','GATX','LSTR','HRI','MOG.A','MOG.B','TKR','CSWI','DY','GXO','BE','HXL','GTES','SNDR','AL',
                         'AMTM','AVAV','KTOS','ACA','SKYW','ACHR','BBU','MATX','EXPO','TNET','FELE','ROAD','MSM','IESC','KAI','STRL','VRRM','AEIS','MMS','CBZ','GEO','RXO','PRIM','BCO','SPR','UNF','NPO','CPA'],save_as_csv=True)


pyT.generateSMAReport(['GE','CAT','RTX','UNP','HON','BA','DE','ETN','LMT','UPS','RELX','PH','WM','MMM','TT','CTAS','ITW','TRI',
                         'TDG','CP','EMR','NOC','GD','RSG','CNI','FDX','CSX','CARR','PCAR','NSC','CPRT','GWW','JCI','HWM','AXON','URI','CMI','WCN','PWR','VRT','DAL','AME','FAST','VRSK','LHX','ODFL','OTIS','IR','FERG','WAB','UAL','EFX','HEI','FER','ROK','XYL','HEI.A','GPN','DOV','VLTO','RYAAY','HUBB','LII','EME','BLDR','SNA','LUV','WSO.B','WSO','AER','CSL','J','SYM','JBHT','PNR','MAS','GFL','IEX','RBA','BAH','CNH','EXPD','OC','XPO','FIX','ZTO','RKLB','UHAL','GGG','TXT','ACM','SWK','ESLT','CW','POOL','UHAL.B','SAIA','NDSN','CLH','RTO','ITT','CHRW','ALLE','MTZ','LECO','CNM','TFII','AAL','WWD','RBC','ULS','NVT','APG','RRX','BWXT','FTAI','ARMK','AYI','BLD','AIT','TTEK','AOS','CR','CRS','AAON','WMS','PAC','DRS','KNX','MIDD','LTM','WCC','ALK','SARO','MLI','FBIN','GTLS','GNRC','STN','DCI','TTC','FLR','FLS','LPX','ASR','ATI','DLB','TREX','AGCO','HII','OSK','CAE','ERJ','ESAB','LOAR','AZEK','BECN','KBR','FCN','ADT','WTS','SPXC','WSC','JBTM','CWST','R','ZWS','VMI','AWI','RHI','MSA','SITE','JOBY','KEX','SMR','FSS','GATX','LSTR','HRI','MOG.A','MOG.B','TKR','CSWI','DY','GXO','BE','HXL','GTES','SNDR','AL',
                         'AMTM','AVAV','KTOS','ACA','SKYW','ACHR','BBU','MATX','EXPO','TNET','FELE','ROAD','MSM','IESC','KAI','STRL','VRRM','AEIS','MMS','CBZ','GEO','RXO','PRIM','BCO','SPR','UNF','NPO','CPA'])
