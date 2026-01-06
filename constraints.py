# constraints.py
from domiknows.graph.logicalConstrain import (
    V, notL, andL, orL, ifL, atLeastL, atMostL
)

# ---------- 1. SENMANTIC DISCTRTISATION ----------
def _concept_groups(graph):
    """返回分组字典"""
    bl   = graph['BL']; lu   = graph['LU']; mabh = graph['MABH']; far  = graph['FAR']
    bd   = graph['BD']; abf  = graph['ABF']; cl   = graph['CL']; rbs  = graph['RBS']
    hbn  = graph['HBN']; bla  = graph['BLA']; abh = graph['ABH']; hbfr = graph['HBFR']
    ahbh = graph['AHBH']; dhbh = graph['DHBH']; bn = graph['BN']

    central = orL(bl('A'), bl('B'), bl('C'), bl('D'))
    edge = orL(bl('H'), bl('I'), bl('J'))

    mabh_low = orL(mabh('A'), mabh('C'), mabh('D'))
    mabh_med = orL(mabh('E'), mabh('F'), mabh('G'))
    mabh_hig = orL(mabh('H'), mabh('I'), mabh('J'))

    abh_low = orL(abh('A'), abh('B'), abh('C'))
    abh_med = orL(abh('D'), abh('E'), abh('F'))
    abh_hig = orL(abh('G'), abh('H'), abh('I'), abh('J'))

    far_low = orL(far('A'), far('C'), far('D'))
    far_med = orL(far('E'), far('F'), far('G'))
    far_hig = orL(far('H'), far('I'), far('J'))

    bd_low = orL(bd('A'), bd('C'), bd('D'))
    bd_med = orL(bd('E'), bd('F'), bd('G'))
    bd_hig = orL(bd('H'), bd('I'), bd('J'))

    abf_low = orL(abf('A'), abf('C'), abf('D'))
    abf_med = orL(abf('E'), abf('F'), abf('G'))
    abf_hig = orL(abf('H'), abf('I'), abf('J'))

    cl_low = orL(cl('A'), cl('C'), cl('D'))
    cl_med = orL(cl('E'), cl('F'), cl('G'))
    cl_hig = orL(cl('H'), cl('I'), cl('J'))

    rbs_low = orL(rbs('A'), rbs('C'), rbs('D'))
    rbs_med = orL(rbs('E'), rbs('F'), rbs('G'))
    rbs_hig = orL(rbs('H'), rbs('I'), rbs('J'))

    hbfr_low = orL(hbfr('A'), hbfr('F'))
    hbfr_med = orL(hbfr('G'))
    hbfr_hig = orL(hbfr('H'))

    hbn_low = orL(hbn('A'), hbn('F'), hbn('G'))
    hbn_med = orL(hbn('H'))
    hbn_hig = orL(hbn('I'), hbn('J'))

    bn_low = orL(bn('A'), bn('C'), bn('D'))
    bn_med = orL(bn('E'), bn('F'), bn('G'))
    bn_hig = orL(bn('H'), bn('I'), bn('J'))

    ahbh_low = orL(ahbh('A'), ahbh('F'), ahbh('G'))
    ahbh_hig = orL(ahbh('H'), ahbh('I'), ahbh('J'))

    dhbh_low = orL(dhbh('A'), dhbh('G'))
    dhbh_hig = orL(dhbh('H'), dhbh('I'), dhbh('J'))

    residential = lu('R')
    commercial  = lu('B')
    water       = lu('E')
    green       = lu('G')
    industrial  = lu('M')
    public      = lu('A')
    utility     = lu('U')

    bla_small = orL(bla('A'), bla('B'), bla('C'))
    bla_med = orL(bla('D'), bla('E'), bla('F'))
    bla_large = orL(bla('G'), bla('H'), bla('I'), bla('J'))

    return dict(
        central=central, edge=edge,
        
        MABH_low=mabh_low, MABH_med=mabh_med, MABH_hig=mabh_hig,
        ABH_low=abh_low, ABH_med=abh_med, ABH_hig=abh_hig,
        AHBH_low=ahbh_low, AHBH_hig=ahbh_hig,
        DHBH_low=dhbh_low, DHBH_hig=dhbh_hig,
        
        BD_low=bd_low, BD_med=bd_med, BD_hig=bd_hig,
        FAR_low=far_low, FAR_med=far_med, FAR_hig=far_hig,
        
        ABF_low=abf_low, ABF_med=abf_med, ABF_hig=abf_hig,
        CL_low=cl_low, CL_med=cl_med, CL_hig=cl_hig,
        RBS_low=rbs_low, RBS_med=rbs_med, RBS_hig=rbs_hig,
        HBFR_low=hbfr_low, HBFR_med=hbfr_med, HBFR_hig=hbfr_hig,
        
        BN_low=bn_low, BN_med=bn_med, BN_hig=bn_hig,
        HBN_low=hbn_low, HBN_med=hbn_med, HBN_hig=hbn_hig,
        
        residential=residential, commercial=commercial, water=water, 
        green=green, industrial=industrial, public=public, utility=utility,
        
        BLA_small=bla_small, BLA_med=bla_med, BLA_large=bla_large,
    )

# ---------- 2. PLANNING COMMON SENSE ----------
def add_spatial_common_sense(graph):
    """Add space common sense constraints"""
    print('Adding space common sense constraints...')
    g = _concept_groups(graph)

    ifL(andL(g['MABH_hig'], g['BD_hig']), 
        atLeastL(g['FAR_hig'], 1), p=75)
    
    ifL(g['ABH_hig'], 
        atLeastL(orL(g['FAR_med'], g['FAR_hig']), 1), p=70)
    
    ifL(g['MABH_hig'], 
        atMostL(g['FAR_low'], 0), p=80)

    ifL(g['FAR_hig'], 
        atMostL(g['BD_low'], 0), p=75)

    ifL(g['MABH_hig'], 
        atMostL(g['ABH_low'], 0), p=75)

    ifL(g['BN_hig'], 
        atLeastL(orL(g['BD_med'], g['BD_hig']), 1), p=70)

    ifL(g['RBS_hig'], 
        atLeastL(g['CL_hig'], 1), p=65)

    ifL(g['ABH_hig'], 
        atLeastL(orL(g['HBN_med'], g['HBN_hig']), 1), p=70)

    ifL(andL(g['commercial'], g['MABH_hig']), andL(
        atLeastL(g['FAR_hig'], 1, p=70),
        atLeastL(g['RBS_hig'], 1, p=65),
        atLeastL(g['CL_hig'], 1, p=60)
    ))

    ifL(andL(g['commercial'], g['MABH_low']), andL(
        atLeastL(g['BD_hig'], 1, p=65),
        atLeastL(g['RBS_hig'], 1, p=70),
        atMostL(g['ABF_hig'], 0, p=60)
    ))

    green_water = orL(g['green'], g['water'])
    ifL(green_water, andL(
        atLeastL(g['FAR_low'], 1, p=80),
        atLeastL(g['HBN_low'], 1, p=80),
        atLeastL(g['MABH_low'], 1, p=80),
        atLeastL(g['HBFR_low'], 1, p=75),
        atLeastL(g['ABH_low'], 1, p=80),
        atLeastL(g['AHBH_low'], 1, p=75),
        atLeastL(g['BD_low'], 1, p=80),
        atLeastL(g['RBS_low'], 1, p=75),
        atLeastL(g['BN_low'], 1, p=80)
    ))

    ifL(g['industrial'], andL(
        atLeastL(g['MABH_low'], 1, p=70),
        atLeastL(g['ABH_low'], 1, p=70),
        atLeastL(g['BD_hig'], 1, p=65)
    ))

    ifL(g['residential'], andL(
        atLeastL(g['BN_hig'], 1, p=70),
        atLeastL(g['DHBH_low'], 1, p=65)
    ))

    print('Done！')
    return graph

def add_enhanced_category_constraints(graph):
    """Enhanced category association constraints"""
    print('Adding category association constraints...')
    g = _concept_groups(graph)
    

    ifL(g['central'], andL(
        atLeastL(g['MABH_hig'], 2),  
        atMostL(g['MABH_low'], 1),   
        atLeastL(g['FAR_hig'], 2),   
        atMostL(g['FAR_low'], 1)     
    ), p=70)
    

    ifL(g['commercial'], andL(
        atLeastL(andL(g['RBS_hig'], g['BD_med']), 1),    
        atMostL(andL(g['ABF_hig'], g['CL_hig']), 1),     
        atLeastL(orL(g['MABH_med'], g['MABH_hig']), 1)   
    ), p=75)
    
    ifL(g['residential'], andL(
        atLeastL(andL(g['BN_hig'], g['BD_med']), 1),     
        atMostL(andL(g['MABH_hig'], g['FAR_low']), 0),   
        atLeastL(g['RBS_med'], 1)                        
    ), p=70)
    

    ifL(orL(g['green'], g['water']), andL(
        atLeastL(andL(g['FAR_low'], g['BD_low']), 2),   
        atMostL(g['MABH_hig'], 0),                     
        atMostL(g['HBN_hig'], 0),                     
        atLeastL(g['ABH_low'], 2)                       
    ), p=80)
    
    ifL(g['industrial'], andL(
        atLeastL(andL(g['BD_hig'], g['MABH_low']), 1),  
        atMostL(g['RBS_hig'], 1),                       
        atLeastL(g['ABH_low'], 1)                       
    ), p=65)
    

    ifL(g['MABH_hig'], andL(
        atLeastL(orL(g['FAR_med'], g['FAR_hig']), 1),    
        atMostL(g['BD_low'], 1)                          
    ), p=75)
    

    ifL(g['RBS_hig'], atLeastL(g['CL_hig'], 1), p=75)
    

    ifL(andL(g['MABH_hig'], g['FAR_low']), 
        atMostL(andL(g['MABH_hig'], g['FAR_low']), 0), p=80)
    
    ifL(andL(g['FAR_hig'], g['BD_low']), 
        atMostL(andL(g['FAR_hig'], g['BD_low']), 0), p=75)
    

    ifL(g['public'], andL(
        atLeastL(andL(g['BD_low'], g['MABH_med']), 1),  
        atMostL(g['FAR_hig'], 1),                        
        atLeastL(g['RBS_med'], 1)                       
    ), p=80)
    

    ifL(g['BLA_large'], andL(
        atLeastL(orL(g['BD_med'], g['BD_low']), 2),     
        atMostL(g['CL_hig'], 1),                        
        atLeastL(g['RBS_med'], 1)                        
    ), p=75)
    

    ifL(g['edge'], andL(
        atLeastL(andL(g['MABH_low'], g['FAR_low']), 2),  
        atMostL(g['ABF_hig'], 1),                       
        atLeastL(g['CL_low'], 1)                        
    ), p=70)

    print('Done！')
    return graph

# ---------- 2. statistical correlation ----------
def add_correlation_constraints(graph):
    """Add correlation constraints between indicators"""
    print('Adding statistical correlation constraints...')
    g = _concept_groups(graph)
    
    # Positive correlation
    positive_correlations = [
        (g['MABH_hig'], g['FAR_hig']),
        (g['MABH_hig'], g['HBN_hig']),
        (g['ABH_hig'], g['FAR_med']),
        
        (g['BD_hig'], g['BN_hig']),
        (g['BD_hig'], g['FAR_hig']),
        
        (g['RBS_hig'], g['CL_hig']),
        (g['RBS_med'], g['CL_med'])
    ]
    
    for pred1, pred2 in positive_correlations:
        ifL(pred1, atLeastL(pred2, 1), p=70)
    
    # negative correlation
    negative_correlations = [
        (g['MABH_hig'], g['BD_low']),
        (g['FAR_hig'], g['BD_low']),
        (g['RBS_low'], g['CL_hig']),
        (g['ABF_hig'], g['BD_hig'])
    ]
    
    for pred1, pred2 in negative_correlations:
        ifL(pred1, atMostL(pred2, 1), p=75)
    
    print('Done！')
    return graph

# ---------- 3. URBAN DESIGN GUILDLINES ----------
def add_spatial_constraints(graph):
    """Adding urban design guildlines"""
    print('Adding urban design guildlines...')
    g = _concept_groups(graph)
    print(f'>>>Initial state, number of knowledge constraints：{len(graph.logicalConstrains)}')
    add_spatial_common_sense(graph)
    print(f'>>>After planning common sense：{len(graph.logicalConstrains)}')
    add_enhanced_category_constraints(graph)
    print(f'>>>After category association：{len(graph.logicalConstrains)}')
    add_correlation_constraints(graph)
    print(f'>>>After statistical correlation：{len(graph.logicalConstrains)}')

    ifL(g['central'], andL(
        atLeastL(g['MABH_hig'], 1, p=85),
        atLeastL(g['FAR_hig'], 1, p=80),
        atLeastL(g['ABH_hig'], 1, p=75)
    ))
    
    ifL(g['edge'], andL(
        atLeastL(g['MABH_low'], 1, p=80),
        atLeastL(g['FAR_low'], 1, p=85),
        atLeastL(g['ABH_low'], 1, p=75)
    ))

    edge_non_eco = andL(g['edge'], notL(orL(g['water'], g['green'])))
    ifL(edge_non_eco, andL(
        atLeastL(g['ABF_low'], 1, p=70),
        atLeastL(g['CL_low'], 1, p=75),
        atLeastL(g['RBS_low'], 1, p=70)
    ))

    ifL(andL(g['residential'], g['MABH_hig']),
        atLeastL(g['CL_low'], 1, p=75))
    
    ifL(andL(g['commercial'], orL(g['MABH_hig'], g['FAR_hig'])), andL(
        atLeastL(g['RBS_low'], 1, p=70),
        atLeastL(g['HBFR_low'], 1, p=75),
        atLeastL(g['ABF_low'], 1, p=70)
    ))

    ifL(andL(g['commercial'], g['ABF_hig']), andL(
        atLeastL(g['BD_med'], 1, p=75),
        atLeastL(g['RBS_low'], 1, p=70),
        atLeastL(g['CL_low'], 1, p=75)
    ))

    ifL(andL(g['residential'], orL(g['ABH_low'], g['MABH_low'])),
        atLeastL(g['FAR_low'], 1, p=80))

    ifL(g['BLA_large'], andL(
        atLeastL(g['BD_low'], 1, p=70),
        atLeastL(g['RBS_low'], 1, p=75),
        atLeastL(g['CL_low'], 1, p=70)
    ))

    ifL(g['commercial'], andL(
        atLeastL(g['HBFR_low'], 1, p=75),
        atLeastL(g['RBS_hig'], 1, p=80)
    ))

    residential_public = orL(g['residential'], g['public'])
    ifL(residential_public, andL(
        atLeastL(g['HBFR_low'], 1, p=80),
        atLeastL(g['RBS_med'], 1, p=75)
    ))

    ifL(g['RBS_low'],
        atLeastL(g['BD_low'], 1, p=75))

    ifL(g['public'], andL(
        atLeastL(orL(g['MABH_low'], g['MABH_med']), 1, p=85),
        atLeastL(g['BD_low'], 1, p=80)
    ))
    
    ifL(g['utility'], andL(
        atLeastL(g['BD_low'], 1, p=90),
        atLeastL(g['FAR_low'], 1, p=85),
        atLeastL(g['HBN_low'], 1, p=80),
        atLeastL(g['ABH_low'], 1, p=85)
    ))
    print(f'>>>Final state：After urban design guildelines：{len(graph.logicalConstrains)}')

    return graph