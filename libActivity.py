'''
A library used for visual activity monitoring.

Author: Cyril Monette
Initial date: 14/11/2025
'''

from RHCVisualisation.libvisu import Hive, thermal_shifts
import cv2, os
import pandas as pd
import numpy as np
from dask import delayed
from libaSensing import Signature, getDTExp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import ttest_rel


def activity(img_slice1, img_slice2, threshold:int, verbose:bool=False):
    assert img_slice1.shape == img_slice2.shape, "Both image slices should have the same shape"
    if verbose:
        print(f"img_slice1 shape: {img_slice1.shape}, img_slice2 shape: {img_slice2.shape}")
    diff = cv2.absdiff(img_slice1, img_slice2)
    # Define an activity mask based on the threshold
    _, activity_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    if activity_mask is None:
        activity_mask = 0
    else:
        activity = cv2.countNonZero(activity_mask)/ activity_mask.size
    return activity


class Activity:
    '''
    Class representing the visual activity metrics for a hive at a specific timestamp.
    Contains activity values for all heaters across the four RPis, as well as aggregated activity per ihl.
    Attributes:
        ts (pd.Timestamp): Timestamp of the activity measurement.
        activity_values (list[dict]): List of 4 dicts, each containing heater activity values for one RPi.
        htr_activity (dict): Aggregated activity per heater for 'upper' and 'lower' ihls.
    '''
    def __init__(self, ts:pd.Timestamp, activity_values:list[dict]):
        '''
        Creates an Activity object.

        Nones in activity_values will spread to yield np.NaN in all heaters in self.htr_activity.
        '''
        assert len(activity_values) == 4, "activity_values must be a list of 4 dicts (one per RPi), or None if no data for that RPi"
        self.ts = ts
        self.activity_values = activity_values  # List with 4 dicts that each have heaters as keys and activity as values
        all_htrs = [f"h{i:02}" for i in range(10)]
        # Fill missing heaters with None activity
        for i, rpi_dict in enumerate(self.activity_values):
            if rpi_dict is None:
                self.activity_values[i] = {htr: None for htr in all_htrs}
            else:
                for htr in all_htrs:
                    if htr not in rpi_dict.keys():
                        self.activity_values[i][htr] = None

        self.htr_activity = self._aggregateActivity()

    def _aggregateActivity(self)->dict:
        '''
        Aggregate the activity values across two RPis for each heater of each ihl.
        
        :return: dict with heater names as keys and aggregated activity as values
        '''
        aggregated_activity = {"upper": {}, "lower":{}}
        for ihl in aggregated_activity.keys():
            # Fill all htrs with 0 to start
            all_htrs = [f"h{i:02}" for i in range(10)]
            for htr in all_htrs:
                aggregated_activity[ihl][htr] = 0.0
            
            rpis = [0,2] if ihl == "upper" else [1,3]
            acts_values = [self.activity_values[rpi] for rpi in rpis]
            for rpi_activity in acts_values:
                for htr, act in rpi_activity.items():
                    if act is None:
                        aggregated_activity[ihl][htr] = None
                    elif aggregated_activity[ihl][htr] is not None:
                        aggregated_activity[ihl][htr] += act

            # Divide by number of RPis to get average
            for htr in aggregated_activity[ihl].keys():
                if aggregated_activity[ihl][htr] is not None:
                    aggregated_activity[ihl][htr] /= len(rpis)

        return aggregated_activity

def computeActivitySingleHtr(hive1:Hive, hive2:Hive, threshold:int, ihl:str, htr:str, verbose:bool=False)->Activity: # TODO: test this function
    '''
    Computes the visual activity between two Hive objects for the specified ihl and heater.

    :param hive1: Hive object at time t1
    :param hive2: Hive object at time t2, which will fix the ts of the activity
    :param threshold: int, pixel difference threshold to consider as activity
    :param ihl: str, either "upper" or "lower"
    :param htr: str, heater number (e.g., "h00", "h01", ..., "h09")
    :return activity: Activity object containing the activity values
    '''
    assert len(hive1.imgs) == len(hive2.imgs) == 4, "Both Hive objects must contain images from 4 RPis"
    assert hasattr(hive1, 'htr_pos') and hasattr(hive2, 'htr_pos'), "Both Hive objects must have heater positions defined"
    assert htr in [f"h{i:02}" for i in range(10)], "htr must be one of 'h00' to 'h09'"
    assert ihl in ['upper', 'lower'], "ihl must be either 'upper' or 'lower'"

    activity_values = []
    rpis = [0,2] if ihl == "upper" else [1,3]
    all_rpis = [0,1,2,3]
    for rpi_idx in all_rpis:
        if rpi_idx not in rpis:
            # This RPi is not considered for this ihl
            activity_values.append(None)
            continue
        if hive1.imgs[rpi_idx] is None or hive2.imgs[rpi_idx] is None:
            # Cannot compute activity for any heater
            activity_values.append(None)
            continue

        pos1 = hive1.htr_pos[rpi_idx][htr]
        pos2 = hive2.htr_pos[rpi_idx][htr]

        # Extract the image slices for the heater positions
        img_slice1 = hive1.imgs[rpi_idx][pos1[0][1]:pos1[1][1], pos1[0][0]:pos1[1][0]]
        img_slice2 = hive2.imgs[rpi_idx][pos2[0][1]:pos2[1][1], pos2[0][0]:pos2[1][0]]

        # Compute activity for the specified heater
        htr_activity = activity(img_slice1, img_slice2, threshold, verbose)
        rpi_activity = {htr: htr_activity}
        activity_values.append(rpi_activity)

    _activity = Activity(ts=hive2.ts, activity_values=activity_values)
    return _activity

def computeActivity(hive1:Hive, hive2:Hive, threshold:int, verbose:bool=False)->Activity: # TODO: test this function
    '''
    Computes the visual activity between two Hive objects for all four RPis images and all heaters within each RPi.

    :param hive1: Hive object at time t1
    :param hive2: Hive object at time t2, which will fix the ts of the activity
    :param threshold: int, pixel difference threshold to consider as activity
    :return activity: Activity object containing the activity values
    '''
    assert len(hive1.imgs) == len(hive2.imgs) == 4, "Both Hive objects must contain images from 4 RPis"
    assert hasattr(hive1, 'htr_pos') and hasattr(hive2, 'htr_pos'), "Both Hive objects must have heater positions defined"

    activity_values = []
    for rpi_idx in range(4):
        if hive1.imgs[rpi_idx] is None or hive2.imgs[rpi_idx] is None:
            # Cannot compute activity for any heater
            activity_values.append(None)
            continue
        rpi_htr_pos1 = hive1.htr_pos[rpi_idx]
        rpi_htr_pos2 = hive2.htr_pos[rpi_idx]
        rpi_activity = {}
        for htr in rpi_htr_pos1.keys():
            pos1 = rpi_htr_pos1[htr]
            pos2 = rpi_htr_pos2[htr]

            # Extract the image slices for the heater positions
            img_slice1 = hive1.imgs[rpi_idx][pos1[0][1]:pos1[1][1], pos1[0][0]:pos1[1][0]]
            img_slice2 = hive2.imgs[rpi_idx][pos2[0][1]:pos2[1][1], pos2[0][0]:pos2[1][0]]

            # Compute activity for this heater
            htr_activity = activity(img_slice1, img_slice2, threshold, verbose)
            rpi_activity[htr] = htr_activity

        activity_values.append(rpi_activity)

    _activity = Activity(ts=hive2.ts, activity_values=activity_values)
    return _activity

@delayed
def computeSignatureActivity(sig:Signature, img_paths:pd.DataFrame, duration:int)->tuple[list[Activity], list[Activity]]:
    '''
    Computes the visual activity for a given signature before and after the signature.
    Warning: This function assumes that image frequency is 1 minute.

    :param sig: Signature object containing the start timestamp
    :param img_paths: DataFrame with image paths indexed by timestamp
    :param duration: int, duration in minutes to consider before and after the signature
    :return: tuple of lists containing (befores_sig_activities, afters_sig_activities)
    '''
    assert duration%2 == 0, "Duration must be an even number."

    befores_sig_activities = []
    afters_sig_activities = []

    exp = getDTExp(sig.ts_start, sig.exp.heater.hive_num)
    all_ts = pd.date_range(start=sig.ts_start_pwm - pd.Timedelta(minutes=duration), end=sig.ts_start_pwm + pd.Timedelta(minutes=duration), freq='1min')
    # Drop seconds from ts
    all_ts = all_ts.map(lambda ts: ts.replace(second=0, microsecond=0))

    activities = []
    prev_hive = None
    for dt in all_ts:            
        ts_images = []
        ts_names = []
        img_path = img_paths.loc[dt].copy()
        for col in img_path.index:
            if img_path[col] is None:
                ts_images.append(None)
                ts_names.append("No image available")
            else:
                img = cv2.imread(img_path[col], cv2.IMREAD_GRAYSCALE)
                ts_images.append(img)
                img_name = img_path[col].split(os.sep)[-1][:-4]
                ts_names.append(img_name)

        _hive = Hive(dt, ts_images, False, ts_names, hive_nb=sig.exp.heater.hive_num)
        _hive.setThermalShifts(thermal_shifts[exp][sig.exp.heater.hive_num])
        if prev_hive is not None:
            activity_metrics = computeActivitySingleHtr(prev_hive, _hive, threshold=25, ihl = sig.exp.heater.ihl, htr=sig.exp.heater.heater_num, verbose=False)
            activities.append(activity_metrics)
        prev_hive = _hive
        del _hive  # Free memory
        del ts_images  # Free memory

    befores_sig_activities = [_act for _act in activities if _act.ts <= sig.ts_start_pwm]
    afters_sig_activities = [_act for _act in activities if _act.ts > sig.ts_start_pwm]
    return befores_sig_activities, afters_sig_activities

def plotActivities(activities:pd.DataFrame, deltaT:float, ihl:str, alternative_ttest = "two-sided", common_y_axis:bool = False, verbose:bool=False):
    '''
    Plot the activity values for a given ihl and deltaT.
    This function assumes that the image frequency is 1min.

    :param activities: df with activities of a specific hive and exp. deltaT, ihl and htr should be columns. 
    sig_activities is the column that contains the tuple of Activity lists as values
    :param ihl: str, either "upper" or "lower"
    :param deltaT: float, the deltaT value to plot
    '''
    assert ihl in ['upper', 'lower'], "ihl must be either 'upper' or 'lower'"
    assert deltaT in [1.0, 3.0, 5.0, -1.0], "deltaT must be one of [1.0, 3.0, 5.0, -1.0]"

    activities = activities.loc[activities['ihl'] == ihl]
    # Check that all sig activities have the same number of activities
    duration = len(activities.iloc[0]['sig_activities'][0])
    for _, row in activities.iterrows():
        assert len(row['sig_activities'][0]) == duration and len(row['sig_activities'][1]) == duration, "All sig activities must have the same number of activities (duration)"
    
    bg_img = cv2.imread(f'/Users/cyrilmonette/Desktop/EPFL 2018-2026/PhD - Mobots/Publishing/Publications/aSensing_IEEEtrans/figures/subfigures/setup.jpeg')
    # Get aspect ratio of bg_img
    aspect_ratio = bg_img.shape[1] / bg_img.shape[0]
    fig = plt.figure(figsize=(20,20/aspect_ratio), constrained_layout=False)
    # fig.patch.set_facecolor("none")
    # fig.patch.set_alpha(0)

    bg_ax = fig.add_axes([0, 0, 1, 1], zorder=-10)
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
    bg_ax.imshow(bg_img, alpha=0.4, aspect='auto')
    bg_ax.axis("off")

    gs = gridspec.GridSpec(4, 7, height_ratios=[0.34, 1, 1, 0.11], width_ratios=[0.4,1,1,1,1,1,0.5])
    # row 0 = blank space for background
    # rows 1 & 2 = the 10 subplots

    # Row 0 stays empty — blank space for background image
    blank_ax = fig.add_subplot(gs[0, :])
    blank_ax.axis("off")   # keep it empty; prevents autoshrink
    # Row 3 stays empty — blank space for background image
    blank_ax = fig.add_subplot(gs[3, :])
    blank_ax.axis("off")   # keep it empty; prevents autoshrink

    # Columns 0 and 6 stay empty — blank space for background image
    for r in [1,2]:
        for c in [0,6]:
            blank_ax = fig.add_subplot(gs[r, c])
            blank_ax.axis("off")   # keep it empty; prevents autoshrink

    # Create the axes in rows 1 and 2
    axes = np.empty((2,5), dtype=object)
    for r in [1,2]:
        for c in range(1,6):
            axes[r-1, c-1] = fig.add_subplot(gs[r, c])

    # Create ax2 for each ax
    axes2 = np.empty((2,5), dtype=object)
    for r in [0,1]:
        for c in range(5):
            axes2[r, c] = axes[r, c].twinx()

    if deltaT == -1.0:
        deltaT = [float(d) for d in activities['deltaT'].unique()]
    else:
        deltaT = [deltaT]

    for dT in deltaT:
        activities_dT = activities.loc[activities['deltaT'] == dT]
        all_htrs = activities_dT['htr'].unique()
        for htr in all_htrs:
            htr_activities = activities_dT.loc[activities_dT['htr']==htr]
            acts_df = pd.DataFrame(columns=['timeDelta','activity'], dtype=float)
            temp_df = None
            pre_sig_avgs = []   # For t-test
            post_sig_avgs = []  # For t-test
            for _, sig_acts in htr_activities.iterrows():
                _sig_acts = sig_acts['sig_activities']
                pre_sig_avgs.append(
                    np.nanmean([
                        act.htr_activity[ihl][htr]
                        if act.htr_activity[ihl][htr] is not None else np.nan
                        for act in _sig_acts[0]
                    ])
                )
                post_sig_avgs.append(
                    np.nanmean([
                        act.htr_activity[ihl][htr]
                        if act.htr_activity[ihl][htr] is not None else np.nan
                        for act in _sig_acts[1]
                    ])
                )

                for j, act in enumerate(_sig_acts[0]):  # before signature
                    acts_df.loc[len(acts_df)] = {'timeDelta': -duration+j+1, 'activity': act.htr_activity[ihl][htr] if act.htr_activity[ihl][htr] is not None else np.nan}

                for j, act in enumerate(_sig_acts[1]):  # after signature
                    acts_df.loc[len(acts_df)] = {'timeDelta': j+1, 'activity': act.htr_activity[ihl][htr] if act.htr_activity[ihl][htr] is not None else np.nan}
                
                _tmp_df = sig_acts['htr_tmp']
                if temp_df is None:
                    temp_df = _tmp_df
                else:
                    temp_df = pd.concat([temp_df, _tmp_df], ignore_index=True)

            acts_df['activity']=acts_df['activity']*100
            # Ensure equal length for paired t-test
            assert len(pre_sig_avgs) == len(post_sig_avgs), "Pre and post values must have the same length for paired t-test"
            if verbose:
                print(f"Heater {htr} ({ihl}): Number of signatures = {len(pre_sig_avgs)}")
            
            # Perform paired t-test
            t_stat, p_val = ttest_rel(pre_sig_avgs, post_sig_avgs, alternative=alternative_ttest) # Use alternative = 'less' to test if post > pre
            if verbose:
                print(f"Heater {htr} ({ihl}):  t={t_stat:.3f},  p={p_val:.3e}")
            # h00 is at ax[0,4], h02 is at ax[0,3], h04 is at ax[0,2], etc.
            ax_row = 0 if int(htr[1:])%2 == 0 else 1
            ax_col = (int(htr[1:]) // 2)
            ax = axes[ax_row, ax_col]
            #sns.lineplot(data=acts_df, x='timeDelta', y='activity', ax=ax, label=f"{htr} activity at {dT}°C", errorbar="sd", estimator='median')
            _label = f"Activity ({dT}°C)" if len(deltaT) > 1 else "Activity"
            sns.lineplot(data=acts_df, x='timeDelta', y='activity', ax=ax, label=_label, errorbar=('pi', 70), estimator='median', err_kws={'alpha':0.15})

            # On the other y-axis, plot the heater temperature
            ax2 = axes2[ax_row, ax_col]
            _label = f"Temp. ({dT}°C)" if len(deltaT) > 1 else "Temp."
            sns.lineplot(data=temp_df, x='timeDelta', y='htr_tmp', ax=ax2, label=_label, linestyle='--', errorbar=('pi', 70), estimator='median', err_kws={'alpha':0.1})

            # ax2.set_ylabel("")
            # ax2.set_yticks([]) # Hide y-ticks for heater temp axis
        
            #ax2.set_ylim(-0.5, 21)
            if dT == deltaT[-1]:
                ax2.set_ylim(-0.5, 21)
                ax2.legend().set_visible(False)
                if ax_col == 4:
                    ax2.set_ylabel("Heater temperature (°C)", fontsize=14)
                    y_ticks = np.arange(0, 5.1, 1)
                    ax2.set_yticks(y_ticks)
                    ax2.tick_params(axis='y', labelsize=12)
                #ax2.legend(loc=(0.02, 0.06))
                # if ax_row == 0 and ax_col == 4:
                #     ax2.legend(loc =(0.025, 0.06))
                # else:
                else:
                    ax2.set_ylabel("")
                    ax2.set_yticks([]) # Hide y-ticks for heater temp axis

            # Hide labels by default
            if ax_col != 0:
                ax.set_ylabel("")
            else:
                ax.set_ylabel("Activity (%)", fontsize=14)

            if ax_row == 0:
                ax.set_xlabel("")
                ax.set_xticks([]) # Hide xticks
            else:
                ax.set_xlabel("Time (minutes)", fontsize=14)
                ax.set_xticks(np.arange(-duration,duration+1,10))

            if dT == deltaT[-1]:
                ax.axvline(x=0, color='r', linestyle='--')
                #ax.axhline(y=0, color='black', linewidth=1, linestyle='--')
                # Annotate htr bottom right
                ax.text(0.95, 0.07, htr, transform=ax.transAxes, fontsize=18,
                    verticalalignment='bottom', horizontalalignment='right', alpha=0.6, fontweight='bold')

            if common_y_axis:
                ax.set_yticks([]) # Will be set later globally

            if dT == deltaT[-1]:
                # Set x-ticks and y-ticks font size
                ax.tick_params(axis='x', labelsize=12)
                ax.tick_params(axis='y', labelsize=12)
                if ax_row == 0 and ax_col == 4:
                    # Add ax2 legend entries to ax legend
                    handles2, labels2 = ax2.get_legend_handles_labels()
                    handles1, labels1 = ax.get_legend_handles_labels()
                    all_handles = handles1 + handles2
                    all_labels = labels1 + labels2
                    ax.legend(all_handles, all_labels, loc ='upper right', fontsize=10)
                    #ax.legend(loc ='upper left')
                else:
                    ax.legend().set_visible(False)

    if common_y_axis:
        # Find global y-limits
        y_mins = []
        y_maxs = []
        for r in range(2):
            for c in range(5):
                ax = axes[r, c]
                ylims = ax.get_ylim()
                y_mins.append(ylims[0])
                y_maxs.append(ylims[1])
        global_ymin = -10
        global_ymax = 31.15 #max(y_maxs)
        # Set all axes to global y-limits
        for r in range(2):
            for c in range(5):
                ax = axes[r, c]
                ax.set_ylim(global_ymin, global_ymax)

        # Add y-ticks to first column
        for r in range(2):
            ax = axes[r, 0]
            # I want integer yticks from global_ymin to global_ymax that cover all multiples of 5 within that range
            y_ticks = np.arange(0, np.ceil(global_ymax/5)*5 + 1, 5)
            # Limit y_ticks to be within global_ymin and global_ymax
            y_ticks = y_ticks[(y_ticks >= global_ymin) & (y_ticks <= global_ymax)]
            ax.set_yticks(y_ticks)
                                      
    # Add a background image to the whole image
    # Rescale bg_img to fig size
    # fig_width, fig_height = fig.get_size_inches() * fig.dpi
    # bg_img = cv2.resize(bg_img, (int(fig_width), int(fig_height)))
    # bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
    # fig.figimage(bg_img, xo=0, yo=0, alpha=0.4, zorder=-1)

    # Make a title on top of all subplots without changing the layout
    gs_title = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0, :])
    title_ax = fig.add_subplot(gs_title[0, 0])
    title_ax.axis("off")
    _text = f"Visual Activities for {ihl} frame at "+r"$\Delta T$" f"={deltaT[0]}°C" if len(deltaT) == 1 else f"Visual Activities for {ihl} frame"
    title_ax.text(0.5, 0.45, _text, fontsize=24, color='black', ha='center', va='center', fontweight='bold')

    if not common_y_axis:
        fig.tight_layout(pad=0.3, h_pad=1.3)
    else:
        fig.subplots_adjust(
            left=0.01,   # enough for y-ticks on leftmost column
            right=0.99,
            top=0.95,
            bottom=0,
            wspace=0.06,
            hspace=0.06
        )

    plt.show()
    return fig