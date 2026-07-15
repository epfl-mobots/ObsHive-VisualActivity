'''
A library used for visual activity monitoring.

Author: Cyril Monette
Initial date: 14/11/2025
'''

from RHCVisualisation.libvisu import Hive
from RHCVisualisation.RHCImaging.libimage import RPiCamV3_img_shape
from typing import List, Dict, Tuple
import cv2, os
import pandas as pd
import numpy as np
from dask import delayed, compute
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import ttest_rel
from abc import ABC, abstractmethod


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
    return activity, activity_mask

class Activity(ABC):
    '''
    Abstract Base Class representing the visual activity metrics for a hive at a specific timestamp.
    '''
    
    def __init__(self, ts:pd.Timestamp):
        self.ts = ts

    @abstractmethod
    def _aggregateActivity(self):
        pass

class RpisActivity(Activity):
    '''
    Activity class representing the visual activity metrics for the four RPis.
    Contains activity values for each RPi, as well as aggregated activity per ihl and for the whole hive.
    Attributes:
        ts (pd.Timestamp): Timestamp of the activity measurement.
        activity_values (list[float]): List of 4 activity values for each RPi.
        ihl_activity (dict): Aggregated activity per ihl for 'upper' and 'lower'.
        hive_activity (float): Aggregated activity for the whole hive.
    '''

    def __init__(self, ts:pd.Timestamp, activity_values:List[float]):
        '''
        Creates an RpisActivity object.

        Nones in activity_values will spread to yield np.NaN in ihl_activity and hive_activity.
        '''
        assert len(activity_values) == 4, "activity_values must be a list of 4 floats (one per RPi), or None if no data for that RPi"
        super().__init__(ts)
        self.activity_values = activity_values  # List with 4 activity values for each RPi
        self.ihl_activity = self._aggregateActivity()
        self.hive_activity = self._aggregateHiveActivity()

    def _aggregateActivity(self)->dict:
        '''
        Aggregate the activity values across both RPis for each ihl.
        
        :return: dict with heater names as keys and aggregated activity as values
        '''
        aggregated_activity = {"upper": 0.0, "lower":0.0}
        for ihl in aggregated_activity.keys():
            rpis = [0,2] if ihl == "upper" else [1,3]
            acts_values = [self.activity_values[rpi] for rpi in rpis]
            if any(act is None for act in acts_values):
                aggregated_activity[ihl] = None
            else:
                aggregated_activity[ihl] = np.mean(acts_values)

        return aggregated_activity
    
    def _aggregateHiveActivity(self)->float:
        '''
        Aggregate the activity values across both IHLs for the whole hive.
        
        :return: float representing the aggregated activity for the whole hive
        '''
        if any(act is None for act in self.activity_values):
            return None
        else:
            return np.mean(self.activity_values)

class HtrsActivity(Activity):
    '''
    Activity class representing the visual activity metrics for all the heaters of the hive.
    Contains activity values for all heaters across the four RPis, as well as aggregated activity per ihl.
    Attributes:
        ts (pd.Timestamp): Timestamp of the activity measurement.
        activity_values (list[dict]): List of 4 dicts, each containing heater activity values for one RPi.
        htr_activity (dict): Aggregated activity per heater for 'upper' and 'lower' ihls.
    '''
    
    HEATER_IDS = [f'h{i:02d}' for i in range(10)]

    def __init__(self, ts:pd.Timestamp, activity_values:List[Dict]):
        '''
        Creates an HtrsActivity object.

        Nones in activity_values will spread to yield np.NaN in all heaters in self.htr_activity.
        '''
        assert len(activity_values) == 4, "activity_values must be a list of 4 dicts (one per RPi), or None if no data for that RPi"
        super().__init__(ts)
        self.activity_values = activity_values  # List with 4 dicts that each have heaters as keys and activity as values
        # Fill missing heaters with None activity
        for i, rpi_dict in enumerate(self.activity_values):
            if rpi_dict is None:
                self.activity_values[i] = {htr: None for htr in HtrsActivity.HEATER_IDS}
            else:
                for htr in HtrsActivity.HEATER_IDS:
                    if htr not in rpi_dict.keys():
                        self.activity_values[i][htr] = None

        self.htr_activity = self._aggregateActivity()

    def _aggregateActivity(self)->dict:
        '''
        Aggregate the activity values across both RPis for each heater of each ihl.
        
        :return: a dict for each IHL with heater names as keys and aggregated activity as values
        '''
        aggregated_activity = {"upper": {}, "lower":{}}
        for ihl in aggregated_activity.keys():
            # Fill all htrs with 0 to start
            for htr in HtrsActivity.HEATER_IDS:
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

@delayed
def computeRpiActivity(img_paths:pd.DataFrame, threshold:int, compute_diff_hives:bool=False, verbose:bool=False)->Tuple[RpisActivity, Hive]:
    '''
    Computes the visual activity (RpisActivity) between TWO timestamps for a given hive and threshold.

    :param img_paths: DataFrame with timestamps as index, 4 columns corresponding to the 4 RPis and two rows corresponding to both timestamps.
    :param threshold: int, pixel difference threshold to consider as activity
    :param compute_diff_hives: bool, whether to compute and return the Hive object representing the difference between consecutive timestamps.
    :return activity: tuple of (RpisActivity object, hive_diff Hive object) representing the differences between consecutive timestamps
    '''

    assert len(img_paths.columns) == 4, "img_paths must have 4 columns corresponding to the 4 RPis"
    assert len(img_paths) == 2, "img_paths must have 2 rows corresponding to the two timestamps to compare"
    assert type(img_paths.index[0]) == pd.Timestamp and type(img_paths.index[1]) == pd.Timestamp, "The index of img_paths must be of type pd.Timestamp"
    assert img_paths.index[0] < img_paths.index[1], "The first row of img_paths should correspond to the earlier timestamp (t1) and the second row to the later timestamp (t2)"
    hive_nb = int(img_paths.columns[0][1])  # Extract hive number from column name (assuming format "h{hive_nb}r{rpi_nb}")
    assert all(col.startswith(f"h{hive_nb}r") for col in img_paths.columns), "All columns in img_paths must correspond to the same hive number and be in the format 'h{hive_nb}r{rpi_nb}'"

    imgs1 = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) if p is not None else None for p in img_paths.iloc[0]]
    imgs2 = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) if p is not None else None for p in img_paths.iloc[1]]

    img_names1 = [img_paths.iloc[0][col].split(os.sep)[-1][:-4] if img_paths.iloc[0][col] is not None else None for col in img_paths.columns]
    img_names2 = [img_paths.iloc[1][col].split(os.sep)[-1][:-4] if img_paths.iloc[1][col] is not None else None for col in img_paths.columns]

    hive1 = Hive(img_paths.index[0], imgs1, False, img_names1, hive_nb=hive_nb)
    hive2 = Hive(img_paths.index[1], imgs2, False, img_names2, hive_nb=hive_nb)

    unique_imgs_1 = hive1.getUniqueRPiImages()
    unique_imgs_2 = hive2.getUniqueRPiImages()
    if verbose:
        print(f"Unique images for hive1 at {hive1.ts}: {[img.shape for img in unique_imgs_1]}")
        # Check if they are the same
        for i in range(4):
            # Check the shape first
            if unique_imgs_1[i].shape != hive1.imgs[i].shape:
                print(f"RPi {i} - unique image and pp image have different shapes: {unique_imgs_1[i].shape} vs {hive1.imgs[i].shape}")
            elif unique_imgs_1[i] is not None and hive1.imgs[i] is not None:
                print(f"RPi {i} - unique image and pp image are the same: {(unique_imgs_1[i] == hive1.imgs[i]).all()}")
            else:
                print(f"RPi {i} - unique image or pp image is None")

    activity_values = []
    if compute_diff_hives:
        activity_masks = []
    
    for unique_img1, unique_img2 in zip(unique_imgs_1, unique_imgs_2):
        if unique_img1 is None or unique_img2 is None:
            activity_values.append(None)
            if compute_diff_hives:
                activity_masks.append(None)
        else:
            act, act_mask = activity(unique_img1, unique_img2, threshold, verbose)
            activity_values.append(act)
            if compute_diff_hives:
                activity_masks.append(act_mask)

    _activity = RpisActivity(hive2.ts, activity_values)

    if compute_diff_hives:    
        extended_activity_masks = []
        for i, act_mask in enumerate(activity_masks):
            if act_mask is None:
                extended_activity_masks.append(None)
                continue
            if i in [0,2]:  # rpis 0 and 2 need padding on the bottom
                pad_height = RPiCamV3_img_shape[0] - act_mask.shape[0]
                extended_mask = np.pad(act_mask, ((0, pad_height), (0, 0)), mode='constant', constant_values=0)
            else:  # rpis 1 and 3 need padding on the top
                pad_height = RPiCamV3_img_shape[0] - act_mask.shape[0]
                extended_mask = np.pad(act_mask, ((pad_height, 0), (0, 0)), mode='constant', constant_values=0)
            extended_activity_masks.append(extended_mask)
        hive_diff = Hive(hive2.ts, extended_activity_masks, True, hive2.imgs_names, hive_nb=hive2.hive_nb)
    else:
        hive_diff = None
    
    return _activity, hive_diff

def computeActivitySingleHtr(hive1:Hive, hive2:Hive, threshold:int, ihl:str, htr:str, verbose:bool=False)->HtrsActivity:
    '''
    Computes the visual activity between two Hive objects for the specified ihl and heater.

    :param hive1: Hive object at time t1
    :param hive2: Hive object at time t2, which will fix the ts of the activity
    :param threshold: int, pixel difference threshold to consider as activity
    :param ihl: str, either "upper" or "lower"
    :param htr: str, heater number (e.g., "h00", "h01", ..., "h09")
    :return activity: HtrsActivity object containing the activity values
    '''

    hive1.computePPImgs() # Ensure the preprocessed images are computed for hive1
    hive2.computePPImgs() # Ensure the preprocessed images are computed for hive2
    assert len(hive1.pp_imgs) == len(hive2.pp_imgs) == 4, "Both Hive objects must contain images from 4 RPis"
    assert hasattr(hive1, 'htr_pos') and hasattr(hive2, 'htr_pos'), "Both Hive objects must have heater positions defined"
    assert htr in HtrsActivity.HEATER_IDS, "htr must be one of 'h00' to 'h09'"
    assert ihl in ['upper', 'lower'], "ihl must be either 'upper' or 'lower'"

    activity_values = []
    rpis = [0,2] if ihl == "upper" else [1,3]
    all_rpis = [0,1,2,3]
    for rpi_idx in all_rpis:
        if rpi_idx not in rpis:
            # This RPi is not considered for this ihl
            activity_values.append(None)
            continue
        if hive1.pp_imgs[rpi_idx] is None or hive2.pp_imgs[rpi_idx] is None:
            # Cannot compute activity for any heater
            activity_values.append(None)
            continue

        pos1 = hive1.htr_pos[rpi_idx][htr]
        pos2 = hive2.htr_pos[rpi_idx][htr]

        # Extract the image slices for the heater positions
        img_slice1 = hive1.pp_imgs[rpi_idx][pos1[0][1]:pos1[1][1], pos1[0][0]:pos1[1][0]]
        img_slice2 = hive2.pp_imgs[rpi_idx][pos2[0][1]:pos2[1][1], pos2[0][0]:pos2[1][0]]

        # Compute activity for the specified heater
        htr_activity = activity(img_slice1, img_slice2, threshold, verbose)
        rpi_activity = {htr: htr_activity}
        activity_values.append(rpi_activity)

    _activity = HtrsActivity(ts=hive2.ts, activity_values=activity_values)
    return _activity

def computeRpiActivities(img_paths:pd.DataFrame, threshold:int=25, compute_diff_hives:bool=False, verbose:bool=False)->Tuple[List[RpisActivity], List[Hive]]:
    '''
    Computes the RpisActivity for each timestamp in img_paths, by comparing pixel values across direct successors in the img_paths DataFrame.
    This means that if steps of 1 minute are used in img_paths, the activity will be computed between t and t+1min, for all timestamps in img_paths.

    Also checks the resulting hive activity values for abnormally high outliers and prints a warning listing the median activity value and, for each offending activity, its timestamp and value.

    :param img_paths: DataFrame with timestamps as index and 4 columns corresponding to the 4 RPis, containing the image paths.
    :param threshold: int, pixel difference threshold to consider as activity
    :param compute_diff_hives: bool, whether to compute and return the Hive objects representing the differences between consecutive timestamps. For large datasets, it might lead to memory issues.
    :param verbose: bool, whether to print verbose output
    :return: a tuple containing a list of RpisActivity objects and a list of Hive objects representing the differences between consecutive timestamps
    '''
    assert len(img_paths.columns) == 4, "img_paths must have 4 columns corresponding to the 4 RPis"
    hive_nb = int(img_paths.columns[0][1])  # Extract hive number from column name (assuming format "h{hive_nb}r{rpi_nb}")
    assert all(col.startswith(f"h{hive_nb}r") for col in img_paths.columns), "All columns in img_paths must correspond to the same hive number and be in the format 'h{hive_nb}r{rpi_nb}'"

    tasks = []
    for i in range(1, len(img_paths)):
        pair_df = img_paths.iloc[i-1:i+1]  # 2 consecutive rows
        task = computeRpiActivity(pair_df, threshold, verbose=verbose)
        tasks.append(task)

    # Compute the activites with dask
    output = compute(*tasks)
    activities = []
    if compute_diff_hives:
        diff_hives = []
    else:
        diff_hives = None
    for result in output:
        if result is not None:
            activity, hive_diff = result
            activities.append(activity)
            if compute_diff_hives:
                diff_hives.append(hive_diff)
        else:
            activities.append(None)
            if compute_diff_hives:
                diff_hives.append(None)

    # Warn about abnormally high hive activity values (e.g. due to camera glare, misalignment,
    # or other artifacts), using an IQR-based outlier check.
    valid_activities = [a for a in activities if a is not None and a.hive_activity is not None]
    hive_activities = np.array([a.hive_activity for a in valid_activities])
    if len(hive_activities) > 0:
        median = np.median(hive_activities)
        q1, q3 = np.percentile(hive_activities, [25, 75])
        upper_bound = q3 + 1.5 * (q3 - q1)
        outliers = [(a.ts, a.hive_activity) for a in valid_activities if a.hive_activity > upper_bound]
        if outliers:
            print(f"\033[91mWatch out, {len(outliers)} value(s) were abnormally big compared to all the values (median={median:.4f}, above {upper_bound:.4f}):\033[0m")
            for ts, value in outliers:
                print(f"\033[91m  - {ts}: {value:.4f}\033[0m")

    return activities, diff_hives

def plotActivities(activities:pd.DataFrame, deltaT:float, ihl:str, alternative_ttest = "two-sided", common_y_axis:bool = False, verbose:bool=False):
    '''
    Plot the activity values for a given ihl and deltaT.
    This function assumes that the image frequency is 1min.

    :param activities: df with activities of a specific hive and exp. deltaT, ihl and htr should be columns. 
    sig_activities is the column that contains the tuple of HtrsActivity lists as values
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