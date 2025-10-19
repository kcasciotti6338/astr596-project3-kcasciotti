#old_code.py

'''
All the functions I replaced when implementing jit 
'''

def initialize_packet(stars, L_packet, band):
    """
    Initialize one packet with luminosity-weighted star selection.
    
    Parameters:
    -----------
    stars : list of Star objects
    L_packet : float
        Luminosity carried by this packet (pre-calculated)
    band : str or None
        If specified ('B', 'V', or 'K'), run for single band
        If None, sample from all bands (advanced option)
    
    Returns:
    --------
    packet : Photon object
        Initialized with star position, direction, L_packet, band, kappa
    """
        
    Ls = [star.L_band[band] for star in stars]
    
    cdf = np.cumsum(Ls / np.sum(Ls))
    idx = np.searchsorted(cdf, np.random.rand())
    
    return emit_packet_from_star(stars[idx], band, L_packet)

def run_mcrt(stars, grid, bands, n_packets):
    
    """
    Main MCRT simulation loop.
    
    Parameters:
    -----------
    stars : list of Star objects
    grid : Grid object
    bands : list of Band objects or strings ['B', 'V', 'K']
    n_packets : int
        Number of packets PER BAND
    save_every : int
        Save checkpoint every N packets
    
    Returns:
    --------
    results : dict
        Per-band results. Structure: results[band] = {...}
        Should include escape fractions, absorbed/escaped luminosities
    
    CRITICAL: L_packet = L_band_total / n_packets
    where L_band_total is the sum of all stars' luminosities 
    in the CURRENT BAND ONLY (not all bands combined).
    
    Note: Reset grid.L_absorbed between bands!
    """
    
    results = { 'B': {'escape fraction': 0, 'absorbed luminosity': 0, 'escaped luminosity': 0}, 
                'V': {'escape fraction': 0, 'absorbed luminosity': 0, 'escaped luminosity': 0}, 
                'K': {'escape fraction': 0, 'absorbed luminosity': 0, 'escaped luminosity': 0} }
    
    tracker = EscapeTracker(bands)
    
    for band in bands:
        Ls = [star.L_band[band] for star in stars]
        L_packet = np.sum(Ls) / n_packets
        
        for i in range(n_packets):
            packet = initialize_packet(stars, L_packet, band)
            result, pos = propagate_packet(packet, grid)
            if result == 'escaped':
                tracker.record_escape(packet)
            elif result == 'absorbed':
                cell = grid.get_cell_indices(pos)
                grid.n_absorbed[cell] += 1
                grid.L_absorbed[cell] += packet.L
    
        results[band]['absorbed luminosity'] = grid.L_absorbed.sum()
       
        #plot_absorption_map(grid, band)
        grid.n_absorbed[:] = 0
        grid.L_absorbed[:] = 0
    
        results[band]['escaped luminosity'] = tracker.L_escaped_by_band[band]
        results[band]['escape fraction'] = tracker.L_escaped_by_band[band] / np.sum(Ls)
    
    return results

def move(self, distance):
    """
    Move photon along its direction.
    
    Parameters:
    -----------
    distance : float
        Distance to move (cm)
    """
    
    self.pos = self.pos + self.dir * distance
    
    pass

def emit_packet_from_star(star, band, L_packet):
    """
    Create packet emitted from stellar surface.
    
    Parameters:
    -----------
    star : Star object
    band : Band object or string
    L_packet : float
        Luminosity carried by packet
    
    Returns:
    --------
    packet : Photon object
    """
    
    dir = sample_isotropic_direction()
    pos = star.pos + star.R * dir
    
    return Photon(pos, dir, L_packet, band, star.id, star.kappa_band[band])

def sample_isotropic_direction():
    """
    Sample random isotropic direction.
    
    Returns:
    --------
    dir_x, dir_y, dir_z : float
        Unit direction vector
    """
    
    u = np.random.uniform(0, 1)
    v = np.random.uniform(0, 1)
    
    theta = np.arccos(2*u - 1)
    phi = 2 * np.pi * v
    
    x_hat = np.sin(theta)*np.cos(phi)
    y_hat = np.sin(theta)*np.sin(phi)
    z_hat = np.cos(theta)
    
    return np.array([x_hat, y_hat, z_hat])

def distance_to_next_boundary(packet, grid):
    
    """
    Calculate distance to next cell boundary.
    
    Parameters:
    -----------
    packet : Photon object
    grid : Grid object
    
    Returns:
    --------
    d_next : float
        Distance to next boundary (cm)
    """
    
    # find min & max corners of cell
    cell_center = ((grid.get_cell_indices(packet.pos) * grid.cell_size) - (grid.L/2) + grid.cell_size/2)
    corner_min = cell_center - grid.cell_size/2
    corner_max = cell_center + grid.cell_size/2
    
    # find distances to each face
    dmin = corner_min - packet.pos
    dmax = corner_max - packet.pos
    d_faces = np.append(dmin, dmax)
    
    # avoid divide by 0 errors
    d = packet.dir
    dir = np.where(d == 0.0, np.inf, d)

    # scale with direction
    dmin_scaled = dmin / dir 
    dmax_scaled = dmax / dir
    d_faces_scaled = np.append(dmin_scaled, dmax_scaled)
    
    # find which face will hit first
    min = np.inf
    for d in d_faces_scaled:
        if d < min and d > 0: 
            min = d

    return min + grid.cell_size[0]/1e6

def propagate_packet(packet, grid):
    """
    Propagate packet through grid until absorbed or escaped.
    
    Parameters:
    -----------
    packet : Photon object
    grid : Grid object
    
    Returns:
    --------
    outcome : str
        'absorbed' or 'escaped'
    location : tuple
        (ix, iy, iz) if absorbed, (x, y, z) if escaped
    """
    u = np.random.uniform(0, 1)
    
    tau_sample = -np.log(u)
    tau_accumulated = 0
    
    while tau_accumulated < tau_sample and grid.is_inside(packet.pos):
        rho_dust = grid.get_dust_density(pos=packet.pos)
        kappa = packet.kappa

        distance = distance_to_next_boundary(packet, grid)
        d_absorb = (tau_sample - tau_accumulated) / (kappa * rho_dust)
        
        if d_absorb <= distance:
            packet.move(d_absorb)
            tau_accumulated = tau_sample
            break
        else:
            tau_accumulated += kappa * rho_dust * distance
            packet.move(distance)
        
    if grid.is_inside(packet.pos) is False:
        return 'escaped', packet.pos
    else: 
        return 'absorbed', packet.pos

def is_inside(self, pos):
    """
    Check if position is inside grid.
    
    Returns:
    --------
    inside : bool
    """
    if np.all(pos >= self.lower_bounds) and np.all(pos <= self.upper_bounds):
        return True
    else:
        return False

def get_cell_indices(self, pos):
    """
    Get cell indices for position.
    
    Returns:
    --------
    ix, iy, iz : int
        Cell indices
    """
    
    if self.is_inside(pos):
        indices = np.floor((pos + self.L/2) / self.cell_size).astype(int)
        return tuple(indices)
    else:
        raise ValueError("Can't return cell indices; packet has escaped.")

def get_dust_density(self, indices=None, pos=None):
    """
    Get dust density in cell.
    
    Returns:
    --------
    rho_dust : float
        Dust density (g/cm^3)
    """

    if indices is not None:
        return self.rho_gas[indices] * self.f_dust_to_gas
    elif pos is not None: 
        indices = self.get_cell_indices(pos)
        return self.rho_gas[indices] * self.f_dust_to_gas
    else: 
        raise ValueError("Must pass either indices or position.")
    




def star_main():
    """
    Star Testing
    """

    filename = '/Users/kcasc/astr596/projects/astr-596-project-03-kcasciotti6338/data/EEM_dwarf_UBVIJHK_colors_Teff.txt'
    data = np.genfromtxt(filename, skip_header=23, max_rows=118, usecols=(0, 1), dtype=[('sp_t', 'U10'), ('T_eff', 'f8')])   
    idx = np.argmin(np.abs(data['T_eff'] - TSUN))
    print('Sun\'s spectral type: ', data['sp_t'][idx])
    
    star = Star(1)
    labels = star.f_strings()
    print(labels['mass'])
    print(labels['T_eff'])
    print(labels['L_bol'])
    print(labels['R'])
    print(labels['id'])
    print(labels['kappa_band'])
    print(labels['L_band'])

    pass

def dust_main():
    """
    Dust Testing
    """

    draine = read_draine_opacity()

    print('B avg opacity: ', calculate_band_averaged_opacity(draine, BANDS['B'][0], BANDS['B'][1], TSUN))

def photon_main():
    """
    Photon Testing
    """
    grid = Grid(4, 4/PC)
    print('L:', grid.L)
    print('cell size:', grid.cell_size)
    print('upper bounds:', grid.upper_bounds)
    
    dir = sample_isotropic_direction()
    pos = np.array([1.5, 1.5, 1.5])
    packet = Photon(pos, dir, 1, 'B', 'test', 1)
    print('position:', packet.pos)
    print('direction:', packet.dir)
    
    state, final_pos = propagate_packet(packet, grid)
    print('state:', state)
    print('final position:', final_pos)
    
def grid_main():
    """
    Grid Testing
    """
    grid = Grid(16, 16/PC)
    print('L:', grid.L)
    print('cell size:', grid.cell_size)
    print('upper bounds:', grid.upper_bounds)
    
    pos = np.array([-1.9, -1.9, -1.9])
    indices = grid.get_cell_indices(pos)
    print('cell indices:', indices)
    print('is inside?', grid.is_inside(pos))
    
    print('dust density:', grid.get_dust_density(indices=indices))
    print('dust density:', grid.get_dust_density(pos=pos))
    
    print('positions', grid.sample_positions(3))

def utils_main():
    """
    Utils Testing
    """
    print('planck:', planck_function(BANDS['B'][0], TSUN))
    
    b = integrate_band(planck_function, BANDS['B'][0], BANDS['B'][1], TSUN)
    print('integrated B band: ', b)


if __name__ == "__main__":
    main()


