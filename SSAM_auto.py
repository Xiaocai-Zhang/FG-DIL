from __future__ import absolute_import
from __future__ import print_function
from pywinauto import application, timings

# from Run_model import result_mod
from alive_progress import alive_bar
import os,csv
import time
Num_of_simulation = 1
debug_mode = False
hide_window = True
#==========================================================================================================================================
# Below are explanatory variables:
# Maximum time-to-collision (TTC) (second)
maxTTC = 5
# Maximum post-encroachment time (PET) (second)
maxPET = 4
#==========================================================================================================================================
# Rear ene angle (degree)
Rear_end_angle = 30
# Crossing angle (degree)   
crossing_angle = 85
# Time for operation to complete (second)    
max_timeout = 50
# Max time to complete analysis(second) 
max_analyse_timeout = 1200
# Interval for wait function assess success condition (second)
access_interval = 0.5
# Reattempt if success condition not meet within wait time
max_attempts = 20

SSAM_exe_path = 'SSAM\SSAM3EXE\SSAM3.exe'
SSAM_exe = os.path.join(os.getcwd(), SSAM_exe_path)


#========================================================================================================
# Run SSAM analysis for every trajectory file in target folder
def Start(EvalOutDir):
    with alive_bar(Num_of_simulation, spinner = 'classic') as bar:
        for file in os.listdir(EvalOutDir):
            if file.endswith(".trj"):
                trj_path = os.path.abspath(os.path.join(EvalOutDir, file))
                trj_ID, conflict_data_path, summary_path = Get_Path(file, EvalOutDir)
                
                if debug_mode: print("Analysing: ", trj_ID)
                app = application.Application(backend="win32")
                for i in range(max_attempts):
                    try:
                        app.start(SSAM_exe)
                        window = app['SSAM3']
                        main_handle = window.element_info.handle
                        window = app.window(title_re="SSAM3", handle = main_handle)
                        if hide_window: window.minimize()
                        window.wait('exists', max_timeout, access_interval)
                        break
                    except timings.TimeoutError as e:
                        print("SSAM3 not open properly") 
                        if i == max_attempts - 1: 
                            raise RuntimeError("reach max attempts") 
                        else:
                            print("Retry: open SSAM3")  
    
                for i in range(max_attempts):
                    try:
                        for i in range(max_attempts):
                            try:    
                                window.Add.click()
                                open_dlg = app['Open']
                                open_dlg.wait('exists', max_timeout, access_interval)
                                if hide_window: open_dlg.minimize()
                                break
                            except timings.TimeoutError as e:
                                print("Open dialog not open properly")
                                if i == max_attempts - 1:
                                    raise RuntimeError("reach max attempts")
                                else:
                                    print("Retry: open add file dialog")  
                 
                        if window.ListBox.item_count() == 0:
                            open_dlg.Edit.set_text(trj_path)
                            
                        for i in range(max_attempts):
                            try:
                                open_dlg[u'&Open'].click()
                                timings.wait_until(max_timeout, access_interval, lambda: open_dlg.exists(), False)
                                break
                            except timings.TimeoutError as e:
                                print("Open dialog not close properly")
                                if i == max_attempts - 1: 
                                    raise RuntimeError("reach max attempts")
                                else:
                                    print("Retry: close add file dialog")  
    
                        timings.wait_until(max_timeout, access_interval, lambda: window.ListBox.item_count(), 1)
                        timings.wait_until(max_timeout, access_interval, lambda: window.ListBox.item_texts()[0], trj_path)
                        break
                    except timings.TimeoutError as e:
                        print("incorrect trj path")
                        if i == max_attempts - 1: 
                            raise RuntimeError("reach max attempts")
                        else:
                            print("Retry: add trj file to list")  
        
                # Modifiy maximum TTC
                Modify_Value(window.Edit, maxTTC)
    
                # Modifiy maximum PET
                Modify_Value(window.Edit2, maxPET)
    
                # Modify rear end angle
                Modify_Value(window.Edit3, Rear_end_angle)
    
                # Modify crossing angle
                Modify_Value(window.Edit4, crossing_angle)
    
                SSAM_Setting_Check(window, trj_path)
                
                # Start analyse trj data in SSAM
                for i in range(max_attempts):
                    try:
                        window.Analyze.click()
                        timings.wait_until(max_analyse_timeout, access_interval, Analyse_Complete_Check, True, str = window.Static7.texts()[0])
                        break
                    except timings.TimeoutError as e:
                        print("Issue when anlysing")
                        if i == max_attempts - 1: 
                            raise RuntimeError("reach max attempts")
                        else:
                            print("Retry: anlyse trj data")
                
                # Close "analyse completed" dialog
                for i in range(max_attempts):
                    try:
                        complete_window_handleID = app.window(title='SSAM3', top_level_only=True, found_index=0).element_info.handle
                        complete_window = app.window(title='SSAM3', handle = complete_window_handleID)
                        complete_window.OK.click()
                        timings.wait_until(max_timeout, access_interval, lambda: complete_window.exists(), False)
                        break
                    except timings.TimeoutError as e:
                        print("Complete window not close properly")
                        if i == max_attempts - 1: 
                            raise RuntimeError("reach max attempts")
                        else:
                            print("Retry: close complete window")
    
                tabc = window.TabControl
        
                # Export summary from SSAM
                for i in range(max_attempts):
                    try:
                        for i in range(max_attempts):
                            try:
                                tabc.select(2)    
                                window.Button2.click()
                                save_dlg = app['Save As']
                                save_dlg.wait('exists', max_timeout, access_interval)
                                if hide_window: save_dlg.minimize()
                                break
                            except timings.TimeoutError as e:
                                print("Save dialog not open properly")
                                if i == max_attempts - 1:
                                    raise RuntimeError("reach max attempts")
                                else:
                                    print("Retry: open save file dialog")
     
                        for i in range(max_attempts):
                            try:
                                save_dlg[u'&Save'].click()
                                timings.wait_until(max_timeout, access_interval, lambda: save_dlg.exists(), False)
                                break
                            except timings.TimeoutError as e:
                                print("Save dialog not close properly")
                                if i == max_attempts - 1: 
                                    raise RuntimeError("reach max attempts")
                                else:
                                    print("Retry: close save file dialog")
                        timings.wait_until(max_timeout, access_interval, lambda: os.path.exists(summary_path))

                        break
                    except timings.TimeoutError as e:
                        print("summary not save properly")
                        if i == max_attempts - 1:
                            raise RuntimeError("reach max attempts")
                        else:
                            print("Retry: save summary file")
        
                # Export conflict data from SSAM
                for i in range(max_attempts):
                    try:
                        for i in range(max_attempts):
                            try:
                                try:
                                    tabc.select(1)
                                except RuntimeError:
                                    window.wait('exists enabled visible ready', max_timeout, access_interval)
                                window.Button2.click()
                                save_dlg = app['Save As']
                                save_dlg.wait('exists', max_timeout, access_interval)
                                if hide_window: save_dlg.minimize()
                                break
                            except timings.TimeoutError as e:
                                print("Save dialog not open properly")
                                if i == max_attempts - 1:
                                    raise RuntimeError("reach max attempts")
                                else:
                                    print("Retry: open save file dialog")
    
                        for i in range(max_attempts):
                            try:
                                save_dlg[u'&Save'].click()
                                timings.wait_until(max_timeout, access_interval, lambda: save_dlg.exists(), False)
                                break
                            except timings.TimeoutError as e:
                                print("Save dialog not close properly")
                                if i == max_attempts - 1: 
                                    raise RuntimeError("reach max attempts")
                                else:
                                    print("Retry: close save file dialog")
                        timings.wait_until(max_timeout, access_interval, lambda: os.path.exists(conflict_data_path))
                        break
                    except timings.TimeoutError as e:
                        print("Conflict data not save properly")
                        if i == max_attempts - 1:
                            raise RuntimeError("reach max attempts")
                        else:
                            print("Retry: save summary file")
    
                app.kill()
                time.sleep(1)
    
                bar()         
#========================================================================================================
def Modify_Value(Edit_key, val):
    Edit_key.set_text(val)
#========================================================================================================
def SSAM_Setting_Check(window, trj_path):
    if str(maxTTC) != window.Edit.window_text():
        raise ValueError("TTC")
    if str(maxPET) != window.Edit2.window_text():
        raise ValueError("PET")
    if str(Rear_end_angle) != window.Edit3.window_text():
        raise ValueError("Rear end Angle")
    if str(crossing_angle) != window.Edit4.window_text():
        raise ValueError("Crossing Angle")
    if  trj_path != window.ListBox.item_texts()[0]:
        raise ValueError("path")
#========================================================================================================
def Analyse_Complete_Check(str):
    return 'Analysis is completed.' in str
#========================================================================================================
def Get_Path(trj_basename, EvalOutDir):
    trj_ID = os.path.splitext(trj_basename)[0]
    conflict_data_basename = trj_ID +'.csv'
    summary_basename = trj_ID + '_summary.csv'
    
    conflict_data_path = os.path.join(EvalOutDir, conflict_data_basename)
    summary_path = os.path.join(EvalOutDir, summary_basename)

    if os.path.exists(conflict_data_path): os.remove(conflict_data_path)
    if os.path.exists(summary_path): os.remove(summary_path)
    return trj_ID, conflict_data_path, summary_path
if __name__ == "__main__":
    print("Running SSAM =====================================================")
    Start()
    print("Output Result ====================================================")
    print("Completed ========================================================")
