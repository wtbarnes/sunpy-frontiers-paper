"""
This script generates the second figure in the paper.
It performs a field extrapolation from HMI synoptic data and overlays the resulting
field lines on observations from AIA, EUI, and EUVI.
"""
import os

import astropy.units as u
import astropy.time
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.gridspec import GridSpec
import numpy as np
from sunpy.net import Fido,attrs
import sunpy.map
from sunpy.coordinates.sun import carrington_rotation_number, carrington_rotation_time
import sunpy_soar  # this registers the SOAR client
import pfsspy
from aiapy.calibrate import correct_degradation, update_pointing


change_obstime = lambda x,y: SkyCoord(x.replicate(observer=x.observer.replicate(obstime=y), obstime=y))


change_obstime_frame = lambda x,y: x.replicate_without_data(observer=x.observer.replicate(obstime=y), obstime=y)


def add_connectors(ax1, ax2, p1, p2, color='k', lw=1):
    con1 = ConnectionPatch(
        (0, 1), ax1.wcs.world_to_pixel(p1), 'axes fraction', 'data', axesA=ax2, axesB=ax1,
        arrowstyle='-', color=color, lw=lw
    )
    con2 = ConnectionPatch(
        (1, 1), ax1.wcs.world_to_pixel(p2), 'axes fraction', 'data', axesA=ax2, axesB=ax1,
        arrowstyle='-', color=color, lw=lw
    )
    ax2.add_artist(con1)
    ax2.add_artist(con2)


if __name__ == '__main__':
    # Get HMI Carrington Data
    car_rot_date = astropy.time.Time('2022-04-01T00:00:00')
    car_rot = carrington_rotation_number(car_rot_date)
    q = Fido.search(
        attrs.Time('2010/01/01', '2010/01/01'),
        attrs.jsoc.Series('hmi.synoptic_mr_polfil_720s'),
        attrs.jsoc.PrimeKey('CAR_ROT', int(car_rot)),
        attrs.jsoc.Notify(os.environ ['JSOC_EMAIL']),
    )
    f = Fido.fetch(q, path='../data/')
    m_hmi = sunpy.map.Map(f)

    # Identify the center of the AR of interest visually from the synoptic magnetogram 
    ar_center = SkyCoord(lon=65*u.deg, lat=15*u.deg, frame=m_hmi.coordinate_frame)
    # We need to correct the obstime of the frame and of the associated observer coordinate.
    # This is because the default obstime of the carrington map is halfway through the carrington rotation. 
    # However, because a synoptic map is comprised of observations from many different times, this is not the obstime for any given slice. 
    # Thus, we look up the obstime associated with our selected longitude and use this to correct our original AR coordinate
    ar_date = carrington_rotation_time(int(car_rot), ar_center.lon)
    ar_center_corrected = change_obstime(ar_center, ar_date)

    # ## Query and Download EUV Data
    aia_or_secchi = ((attrs.Instrument.aia | attrs.Instrument.secchi)
                     & attrs.Wavelength(171*u.angstrom)
                     & attrs.Sample(5*u.minute))
    eui_query = attrs.Level(2) & attrs.soar.Product('EUI-FSI174-IMAGE')
    q = Fido.search(attrs.Time(ar_date-2*u.minute, ar_date+2*u.minute), aia_or_secchi | eui_query)
    files = Fido.fetch(q, path='../data/')
    m_secchi, m_aia, m_eui = sunpy.map.Map(sorted(files))

    # In the case of AIA, we'll correct the pointing keywords and also correct for the instrument degradation
    m_aia = correct_degradation(update_pointing(m_aia))
    # The SECCHI and AIA maps have also not yet been normalized for exposure time
    m_secchi = m_secchi / m_secchi.exposure_time
    m_aia = m_aia / m_aia.exposure_time

    # Create cutouts around ARs
    ar_width = 700*u.arcsec
    ar_height = 700*u.arcsec
    m_cutouts = []
    for m in [m_aia, m_eui, m_secchi]:
        ar_center_corrected_trans = ar_center_corrected.transform_to(m.coordinate_frame)
        blc = SkyCoord(Tx=ar_center_corrected_trans.Tx-ar_width/2,
                       Ty=ar_center_corrected_trans.Ty-ar_height/2,
                       frame=ar_center_corrected_trans)
        # Each map is rotated prior to submapping such that the selection is aligned with the coordinate grid
        m_cutouts.append(m.rotate(missing=0.0).submap(blc, width=ar_width, height=ar_height))

    # Potential field extrapolation
    m_hmi_resample = m_hmi.resample((1080, 540)*u.pix)
    nrho = 70
    rss = 2.5
    pfss_input = pfsspy.Input(m_hmi_resample, nrho, rss)
    pfss_output = pfsspy.pfss(pfss_input)

    # Select only the fieldlines that are within a certain area around the active region
    new_frame = change_obstime_frame(m_hmi.coordinate_frame, m_cutouts[0].date)
    blc_ar_synop = change_obstime(m_cutouts[0].bottom_left_coord.transform_to(new_frame), m_hmi.date)
    trc_ar_synop = change_obstime(m_cutouts[0].top_right_coord.transform_to(new_frame), m_hmi.date)
    # Mask all those points that are above a certain LOS field strength
    masked_pix_y, masked_pix_x = np.where(m_hmi_resample.data < -1e1)
    seeds = m_hmi_resample.pixel_to_world(masked_pix_x*u.pix, masked_pix_y*u.pix,).make_3d()
    in_lon = np.logical_and(seeds.lon > blc_ar_synop.lon, seeds.lon < trc_ar_synop.lon)
    in_lat = np.logical_and(seeds.lat > blc_ar_synop.lat, seeds.lat < trc_ar_synop.lat)
    seeds = seeds[np.where(np.logical_and(in_lon, in_lat))]
    # Trace fieldlines from seeds specified above
    ds = 0.01
    max_steps = int(np.ceil(10 * nrho / ds))
    tracer = pfsspy.tracing.FortranTracer(step_size=ds,max_steps=max_steps)
    fieldlines = tracer.trace(SkyCoord(seeds), pfss_output,)
    # Adjust obstime of all coordinates to coincide with AR at disk center
    fline_coords = [change_obstime(f.coords, m_aia.date) for f in fieldlines.closed_field_lines if f.coords.shape[0]>500]

    # Build final figure
    h_w_ratio = 21 / 18
    width = 12
    frame_color = 'C3'
    fig = plt.figure(figsize=(width, width*h_w_ratio))
    gs = GridSpec(3, 3, figure=fig)
    #Plot HMI synoptic map
    ax = fig.add_subplot(gs[0,:2], projection=m_hmi)
    m_hmi.plot(axes=ax, title='HMI Synoptic Magnetogram')
    m_hmi.draw_quadrangle(blc_ar_synop, top_right=trc_ar_synop, color=frame_color)
    ax.coords[0].grid(color='k')
    ax.coords[1].grid(color='k')
    # Plot spacecraft locations
    ax = fig.add_subplot(gs[0,2],projection='polar')
    ax.plot(0, 0, marker='o', markersize=15, label='Sun', color='yellow')
    for m in m_cutouts:
        sat = m.observatory
        coord = m.observer_coordinate
        ax.plot(coord.lon.to('rad'), coord.radius.to(u.AU), 'o', label=sat)
        ax.text(coord.lon.to_value('rad')*1.15, coord.radius.to_value(u.AU)*0.95, sat)
    ax.set_theta_zero_location("S")
    ax.set_rlabel_position(225)
    ax.set_rlim(0, 1.1)
    # Plot full-disk EUV images
    full_disk_axes = []
    for i,m in enumerate([m_aia,m_eui,m_secchi]):
        ax = fig.add_subplot(gs[1,i],projection=m)
        m.plot(axes=ax,clip_interval=(1,99.99)*u.percent,
            title=f'{m.observatory} {m.detector} {m.wavelength.to_string(format="latex")}')
        m.draw_quadrangle(m_cutouts[i].bottom_left_coord, top_right=m_cutouts[i].top_right_coord, color=frame_color, lw=1)
        if i:
            ax.coords[1].set_axislabel(' ')
        ax.coords[0].set_axislabel(' ')
        full_disk_axes.append(ax)
        ax.coords[1].set_ticklabel(rotation=90)
    # Plot EUV cutouts with fieldines
    for i, m in enumerate(m_cutouts):
        ax = fig.add_subplot(gs[2,i],projection=m)
        m.plot(
            axes=ax,
            title=False,
            clip_interval=(1,99.99)*u.percent,
        )
        bounds = ax.axis()
        for c in fline_coords[::8]:
            ax.plot_coord(c, lw=1, color='C2',alpha=.75)
        ax.axis(bounds)
        if i:
            ax.coords[1].set_axislabel(' ')
        bottom_right = SkyCoord(Tx=m_cutouts[i].top_right_coord.Tx, Ty=m_cutouts[i].bottom_left_coord.Ty, frame=m_cutouts[i].coordinate_frame)
        add_connectors(full_disk_axes[i], ax, m_cutouts[i].bottom_left_coord, bottom_right, color=frame_color, lw=1)
        ax.grid(alpha=0)
        ax.coords[0].set_ticks(direction='in', color=frame_color,)
        ax.coords[1].set_ticks(direction='in', color=frame_color,)
        ax.coords[0].frame.set_color(frame_color)
        ax.coords[0].frame.set_linewidth(1)
        ax.coords[1].frame.set_color(frame_color)
        ax.coords[1].frame.set_linewidth(1)
        ax.coords[1].set_ticklabel(rotation=90)
        
    plt.subplots_adjust(hspace=0.0)
    fig.savefig('loops-multi-viewpoint.pdf', bbox_inches='tight')
