import logging
import os
from typing import Annotated, Optional

import pathlib as plb
import tempfile
import dicom2nifti
import nibabel as nib
import pydicom
import shutil
from tqdm import tqdm

import vtk
import numpy as np
import surfa as sf
import tensorflow as tf
import voxelmorph as vxm
import neurite as ne
import nibabel as nib
import SimpleITK as sitk
import sitkUtils

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode


#
# SynCT
#


class SynCT(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("SynCT")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#SynCT">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        # slicer.app.connect("startupCompleted()", registerSampleData)



#
# SynCTWidget
#


class SynCTWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SynCT.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = SynCTLogic()

        # 关联MRML场景（关键步骤！）
        self.ui.MRMLNodeComboBox_1_skull.setMRMLScene(slicer.mrmlScene)
        self.ui.MRMLNodeComboBox_2_skull.setMRMLScene(slicer.mrmlScene)
        self.ui.MRMLNodeComboBox_3_skull.setMRMLScene(slicer.mrmlScene)

        self.ui.MRMLNodeComboBox_1_CTclip.setMRMLScene(slicer.mrmlScene)
        self.ui.MRMLNodeComboBox_5.setMRMLScene(slicer.mrmlScene)
        self.ui.MRMLNodeComboBox_6.setMRMLScene(slicer.mrmlScene)
        self.ui.MRMLNodeComboBox_2_syn.setMRMLScene(slicer.mrmlScene)
        self.ui.MRMLNodeComboBox_7.setMRMLScene(slicer.mrmlScene)
        self.ui.MRMLNodeComboBox_8.setMRMLScene(slicer.mrmlScene)
        self.ui.MRMLNodeComboBox_9_mapping.setMRMLScene(slicer.mrmlScene)
        self.ui.MRMLNodeComboBox_10_mapping.setMRMLScene(slicer.mrmlScene)
        self.ui.MRMLNodeComboBox_9.setMRMLScene(slicer.mrmlScene)
        self.ui.MRMLNodeComboBox_10.setMRMLScene(slicer.mrmlScene)

        self.ui.MRMLNodeComboBox_1.setMRMLScene(slicer.mrmlScene)
        self.ui.MRMLNodeComboBox_2.setMRMLScene(slicer.mrmlScene)
        self.ui.MRMLNodeComboBox_2_rigid.setMRMLScene(slicer.mrmlScene)
        self.ui.MRMLNodeComboBox_3_rigid.setMRMLScene(slicer.mrmlScene)
        # self.ui.MRMLNodeComboBox_3.setMRMLScene(slicer.mrmlScene)
        # self.ui.MRMLNodeComboBox_4.setMRMLScene(slicer.mrmlScene)
        

        self.ui.MRMLNodeComboBox_1_skull.connect("currentNodeChanged(vtkMRMLNode*)", 
            lambda node: self.onNodeSelected(node, self.logic.loadImage))
        self.ui.MRMLNodeComboBox_2_skull.connect("currentNodeChanged(vtkMRMLNode*)", 
            lambda node: self.onNodeSelected(node, self.logic.loadImage_skull_pet))
        self.ui.MRMLNodeComboBox_3_skull.connect("currentNodeChanged(vtkMRMLNode*)", 
            lambda node: self.onNodeSelected(node, self.logic.loadImage_skull_mask))
        
        self.ui.MRMLNodeComboBox_1_CTclip.connect("currentNodeChanged(vtkMRMLNode*)", 
            lambda node: self.onNodeSelected(node, self.logic.loadImage_CTclip))

        self.ui.MRMLNodeComboBox_1.connect("currentNodeChanged(vtkMRMLNode*)", 
            lambda node: self.onNodeSelected(node, self.logic.loadImage1_1))
        self.ui.MRMLNodeComboBox_2.connect("currentNodeChanged(vtkMRMLNode*)", 
            lambda node: self.onNodeSelected(node, self.logic.loadImage1_2))
        self.ui.MRMLNodeComboBox_2_rigid.connect("currentNodeChanged(vtkMRMLNode*)", 
            lambda node: self.onNodeSelected(node, self.logic.loadImage1_3))
        self.ui.MRMLNodeComboBox_3_rigid.connect("currentNodeChanged(vtkMRMLNode*)", 
            lambda node: self.onNodeSelected_yield(node, self.logic.loadDeformationField))
        
        # self.ui.MRMLNodeComboBox_3.connect("currentNodeChanged(vtkMRMLNode*)", 
        #     lambda node: self.onNodeSelected(node, self.logic.loadImage2_1))
        # self.ui.MRMLNodeComboBox_4.connect("currentNodeChanged(vtkMRMLNode*)", 
        #     lambda node: self.onNodeSelected(node, self.logic.loadImage2_2))
        
        self.ui.MRMLNodeComboBox_5.connect("currentNodeChanged(vtkMRMLNode*)",
            lambda node: self.onNodeSelected(node, self.logic.loadImage3_1))
        self.ui.MRMLNodeComboBox_6.connect("currentNodeChanged(vtkMRMLNode*)",
            lambda node: self.onNodeSelected(node, self.logic.loadImage3_2))
        self.ui.MRMLNodeComboBox_2_syn.connect("currentNodeChanged(vtkMRMLNode*)",
            lambda node: self.onNodeSelected(node, self.logic.loadImage3_3))
        
        self.ui.MRMLNodeComboBox_7.connect("currentNodeChanged(vtkMRMLNode*)",
            lambda node: self.onNodeSelected(node, self.logic.loadImage4_1))
        self.ui.MRMLNodeComboBox_8.connect("currentNodeChanged(vtkMRMLNode*)",
            lambda node: self.onNodeSelected(node, self.logic.loadImage4_2))
        self.ui.MRMLNodeComboBox_9.connect("currentNodeChanged(vtkMRMLNode*)",
            lambda node: self.onNodeSelected(node, self.logic.loadImage5_1))
        self.ui.MRMLNodeComboBox_10.connect("currentNodeChanged(vtkMRMLNode*)",
            lambda node: self.onNodeSelected(node, self.logic.loadImage5_2))
        
        self.ui.MRMLNodeComboBox_9_mapping.connect("currentNodeChanged(vtkMRMLNode*)",
            lambda node: self.onNodeSelected(node, self.logic.loadImage5_1_mapping))
        self.ui.MRMLNodeComboBox_10_mapping.connect("currentNodeChanged(vtkMRMLNode*)",
            lambda node: self.onNodeSelected(node, self.logic.loadImage5_2_mapping))
        

        # 单例图像处理
        self.ui.Dicom2Nifit_CT.connect("clicked(bool)", self.onDicom2Nifit_CT)
        self.ui.Dicom2Nifit_PET.connect("clicked(bool)", self.onDicom2Nifit_PET)
        self.ui.skullStripButton.connect("clicked(bool)", self.onSkullStrip)
        self.ui.skullStripButton_mask.connect("clicked(bool)", self.onSkullStrip_mask)
        self.ui.ctclipButton.connect("clicked(bool)", self.onCTclip)
        self.ui.rigidRegisterButton.connect("clicked(bool)", self.onRigidRegister)
        self.ui.rigidRegisterButton_field.connect("clicked(bool)", self.onRigidRegister_field)
        # self.ui.spaceRegisterButton.connect("clicked(bool)", self.onSpaceRegister)
        self.ui.synRegisterButton.connect("clicked(bool)", self.onSynRegister)
        self.ui.synRegisterButton_field.connect("clicked(bool)", self.onSynRegister_field)
        self.ui.diceCalculateButton.connect("clicked(bool)", self.onDiceCompute)
        self.ui.SUVrCalculateButton.connect("clicked(bool)", self.onSuvrCompute)
        self.ui.SUVrMappingButton.connect("clicked(bool)", self.onSuvrMapping)

        # 批量CT dicom2nifit
        self.ui.bachApplyButton8.connect("clicked(bool)", self.onDialogShow8)
        self.ui.lineEdit_8_1.connect('editingFinished()', self.onPath8_1Edited)
        self.ui.cancelButton8.connect("clicked(bool)", self.onCancel8)
        self.ui.applyButton8.connect("clicked(bool)", self.onApplyClicked8)

        # 批量PET dicom2nifit
        self.ui.bachApplyButton9.connect("clicked(bool)", self.onDialogShow9)
        self.ui.lineEdit_9_1.connect('editingFinished()', self.onPath9_1Edited)
        self.ui.cancelButton9.connect("clicked(bool)", self.onCancel9)
        self.ui.applyButton9.connect("clicked(bool)", self.onApplyClicked9)

        # 批量CTClip
        self.ui.bachApplyButton7.connect("clicked(bool)", self.onDialogShow7)
        self.ui.lineEdit_7_1.connect('editingFinished()', self.onPath7_1Edited)
        self.ui.cancelButton7.connect("clicked(bool)", self.onCancel7)
        self.ui.applyButton7.connect("clicked(bool)", self.onApplyClicked7)

        # 批量头骨剥离
        self.ui.bachApplyButton6.connect("clicked(bool)", self.onDialogShow6)
        self.ui.lineEdit_6_1.connect('editingFinished()', self.onPath6_1Edited)
        self.ui.cancelButton6.connect("clicked(bool)", self.onCancel6)
        self.ui.applyButton6.connect("clicked(bool)", self.onApplyClicked6)

        self.ui.bachApplyButton6_pet.connect("clicked(bool)", self.onDialogShow6_pet)
        self.ui.lineEdit_6_1_pet.connect('editingFinished()', self.onPath6_1Edited_pet)
        self.ui.cancelButton6_pet.connect("clicked(bool)", self.onCancel6_pet)
        self.ui.applyButton6_pet.connect("clicked(bool)", self.onApplyClicked6_pet)

        # 批量刚性配准
        self.ui.bachApplyButton1.connect("clicked(bool)", self.onDialogShow1)
        self.ui.lineEdit_1_1.connect('editingFinished()', self.onPath1_1Edited)
        self.ui.cancelButton1.connect("clicked(bool)", self.onCancel1)
        self.ui.applyButton1.connect("clicked(bool)", self.onApplyClicked1)

        self.ui.bachApplyButton1_pet.connect("clicked(bool)", self.onDialogShow1_pet)
        self.ui.lineEdit_1_1_pet.connect('editingFinished()', self.onPath1_1Edited_pet)
        self.ui.cancelButton1_pet.connect("clicked(bool)", self.onCancel1_pet)
        self.ui.applyButton1_pet.connect("clicked(bool)", self.onApplyClicked1_pet)

        # 批量空间配准
        # self.ui.bachApplyButton2.connect("clicked(bool)", self.onDialogShow2)
        # self.ui.lineEdit_2_1.connect('editingFinished()', self.onPath2_1Edited)
        # self.ui.cancelButton2.connect("clicked(bool)", self.onCancel2)
        # self.ui.applyButton2.connect("clicked(bool)", self.onApplyClicked2)

        # 批量synthmorph配准
        self.ui.bachApplyButton3.connect("clicked(bool)", self.onDialogShow3)
        self.ui.lineEdit_3_1.connect('editingFinished()', self.onPath3_1Edited)
        self.ui.cancelButton3.connect("clicked(bool)", self.onCancel3)
        self.ui.applyButton3.connect("clicked(bool)", self.onApplyClicked3)

        self.ui.bachApplyButton3_pet.connect("clicked(bool)", self.onDialogShow3_pet)
        self.ui.lineEdit_3_1_pet.connect('editingFinished()', self.onPath3_1Edited_pet)
        self.ui.cancelButton3_pet.connect("clicked(bool)", self.onCancel3_pet)
        self.ui.applyButton3_pet.connect("clicked(bool)", self.onApplyClicked3_pet)

        # 批量Dice计算
        self.ui.bachApplyButton4.connect("clicked(bool)", self.onDialogShow4)
        self.ui.lineEdit_4_1.connect('editingFinished()', self.onPath4_1Edited)
        self.ui.cancelButton4.connect("clicked(bool)", self.onCancel4)
        self.ui.applyButton4.connect("clicked(bool)", self.onApplyClicked4)

        # 批量SUVr Mapping
        self.ui.bachApplyButton10_pet.connect("clicked(bool)", self.onDialogShow10_pet)
        self.ui.lineEdit_10_1_pet.connect('editingFinished()', self.onPath10_1Edited_pet)
        self.ui.cancelButton10_pet.connect("clicked(bool)", self.onCancel10_pet)
        self.ui.applyButton10_pet.connect("clicked(bool)", self.onApplyClicked10_pet)

        # 批量SUVr计算
        self.ui.bachApplyButton5.connect("clicked(bool)", self.onDialogShow5)
        self.ui.lineEdit_5_1.connect('editingFinished()', self.onPath5_1Edited)
        self.ui.cancelButton5.connect("clicked(bool)", self.onCancel5)
        self.ui.applyButton5.connect("clicked(bool)", self.onApplyClicked5)

    # 批量CT dicom2nifit
    def onDialogShow8(self):
        # 显示对话框
        self.ui.dialog8.exec_()

    def onPath8_1Edited(self):
        """当第一个路径编辑框失去焦点时调用"""
        path = self.ui.lineEdit_8_1.text
        if not path:
            return
        
        if not os.path.isdir(path):
            slicer.util.errorDisplay(f"Path does not exist or is not a directory: {path}")
            return
        
        # 获取子文件夹
        try:
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            subdirs.sort()
            
            # 在显示框中显示子文件夹
            self.ui.textEdit_display8.clear()
            if subdirs:
                self.ui.textEdit_display8.append("Subdirectories:")
                for subdir in subdirs:
                    self.ui.textEdit_display8.append(f" - {subdir}")
            else:
                self.ui.textEdit_display8.append("No subdirectories found")
        except Exception as e:
            slicer.util.errorDisplay(f"Error reading directory: {str(e)}")

    def onCancel8(self):
        """取消按钮点击事件"""
        self.ui.dialog8.close()

    def onApplyClicked8(self):
        """Apply按钮点击事件"""
        # 获取输入参数
        base_dir = self.ui.lineEdit_8_1.text.strip()
        fold_name = self.ui.lineEdit_8_2.text.strip()
        output_path = self.ui.lineEdit_8_3.text.strip()
        output_file_name = self.ui.lineEdit_8_4.text.strip()
        
        # 验证输入
        if not all([base_dir, fold_name, output_path, output_file_name]):
            slicer.util.errorDisplay("All fields are required!")
            return
        
        if not os.path.isdir(base_dir):
            slicer.util.errorDisplay(f"Base directory does not exist: {base_dir}")
            return
        
        # 开始处理        
        try:
            # 获取所有子文件夹名
            subdirs = [d for d in os.listdir(base_dir) 
                    if os.path.isdir(os.path.join(base_dir, d))]
            subdirs.sort()
            
            if not subdirs:
                slicer.util.errorDisplay("no subdirs", windowTitle="Error dir path")
                return
            
            # 初始化
            self.ui.progressBar8.setValue(0)
            self.ui.progressBar8.setFormat("Prepare CT dicom2nifit...")
            
            
            # 遍历每个子文件夹
            for i, subdir in enumerate(subdirs):
                self.updateProgress8(10 + i*80, f"Processing {i}/{len(subdirs)} {subdir}...")
                subdir_path = os.path.join(base_dir, subdir)
                print(f"Processing subdir: {subdir_path}")
                
                # 查找指定文件夹
                fold_path = os.path.join(subdir_path, fold_name)
                output_file_path = os.path.join(output_path, subdir)
                
                # 确保输出目录存在
                os.makedirs(output_file_path, exist_ok=True)

                if os.path.isdir(fold_path):
                    self.ui.progressBar8.setMaximum(10 + len(subdirs)*80)  # 分配权重

                    with tempfile.TemporaryDirectory() as tmp_dir:
                        tmp_path = plb.Path(tmp_dir)
                        
                        print("开始DICOM到NIfTI转换...")
                        
                        # 将DICOM目录转换为NIfTI
                        dicom2nifti.convert_directory(fold_path, str(tmp_path), compression=False, reorient=True)
                        
                        nii = next(tmp_path.glob('*nii'))
                        
                        # 修正：使用os.path.join来拼接路径
                        output_file = os.path.join(output_file_path, f'{output_file_name}.nii')
                        shutil.copy(nii, output_file)
                        ct_path = output_file
                        
                        print(f"转换成功: {ct_path}")
                    
                    
            self.updateProgress8(10 + len(subdirs)*80, "Complete CT dicom2nifit!")  
            
        except Exception as e:
            slicer.util.errorDisplay(f"Error during processing: {str(e)}")

    def updateProgress8(self, value, message):
        """更新进度辅助函数"""
        self.ui.progressBar8.setValue(value)
        self.ui.progressBar8.setFormat(message)
        slicer.app.processEvents()

    # 批量PET dicom2nifit
    def onDialogShow9(self):
        # 显示对话框
        self.ui.dialog9.exec_()

    def onPath9_1Edited(self):
        """当第一个路径编辑框失去焦点时调用"""
        path = self.ui.lineEdit_9_1.text
        if not path:
            return
        
        if not os.path.isdir(path):
            slicer.util.errorDisplay(f"Path does not exist or is not a directory: {path}")
            return
        
        # 获取子文件夹
        try:
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            subdirs.sort()
            
            # 在显示框中显示子文件夹
            self.ui.textEdit_display9.clear()
            if subdirs:
                self.ui.textEdit_display9.append("Subdirectories:")
                for subdir in subdirs:
                    self.ui.textEdit_display9.append(f" - {subdir}")
            else:
                self.ui.textEdit_display9.append("No subdirectories found")
        except Exception as e:
            slicer.util.errorDisplay(f"Error reading directory: {str(e)}")

    def onCancel9(self):
        """取消按钮点击事件"""
        self.ui.dialog9.close()

    def onApplyClicked9(self):
        """Apply按钮点击事件"""
        # 获取输入参数
        base_dir = self.ui.lineEdit_9_1.text.strip()
        fold_name = self.ui.lineEdit_9_2.text.strip()
        output_path = self.ui.lineEdit_9_3.text.strip()
        output_file_name = self.ui.lineEdit_9_4.text.strip()
        
        # 验证输入
        if not all([base_dir, fold_name, output_path, output_file_name]):
            slicer.util.errorDisplay("All fields are required!")
            return
        
        if not os.path.isdir(base_dir):
            slicer.util.errorDisplay(f"Base directory does not exist: {base_dir}")
            return
        
        # 开始处理        
        try:
            # 获取所有子文件夹名
            subdirs = [d for d in os.listdir(base_dir) 
                    if os.path.isdir(os.path.join(base_dir, d))]
            subdirs.sort()
            
            if not subdirs:
                slicer.util.errorDisplay("no subdirs", windowTitle="Error dir path")
                return
            
            # 初始化
            self.ui.progressBar9.setValue(0)
            self.ui.progressBar9.setFormat("Prepare PET dicom2nifit...")
            
            
            # 遍历每个子文件夹
            for i, subdir in enumerate(subdirs):
                self.updateProgress9(10 + i*80, f"Processing {i}/{len(subdirs)} {subdir}...")
                subdir_path = os.path.join(base_dir, subdir)
                print(f"Processing subdir: {subdir_path}")
                
                # 查找指定文件夹
                fold_path = os.path.join(subdir_path, fold_name)
                output_file_path = os.path.join(output_path, subdir)
                
                # 确保输出目录存在
                os.makedirs(output_file_path, exist_ok=True)

                if os.path.isdir(fold_path):
                    self.ui.progressBar9.setMaximum(10 + len(subdirs)*80)  # 分配权重

                    fold_path_obj = plb.Path(fold_path)

                    first_pt_dcm = next(fold_path_obj.glob('*'))
                    suv_corr_factor = self.logic.calculate_suv_factor(first_pt_dcm)
                    
                    # 使用临时目录进行转换
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        tmp_path = plb.Path(tmp_dir)
                        
                        print("开始DICOM到NIfTI转换...")
                        
                        # 将DICOM目录转换为NIfTI
                        dicom2nifti.convert_directory(fold_path, str(tmp_path), compression=False, reorient=True)
                        
                        nii = next(tmp_path.glob('*nii'))
                        suv_pet_nii = self.logic.convert_pet(nib.load(nii), suv_factor=suv_corr_factor)
                        output_file = os.path.join(output_file_path, f'{output_file_name}.nii')
                        nib.save(suv_pet_nii, output_file)
                        pet_path = output_file
                        
                        print(f"转换成功: {pet_path}")
                    
                    
            self.updateProgress9(10 + len(subdirs)*80, "Complete PET dicom2nifit!")  
            
        except Exception as e:
            slicer.util.errorDisplay(f"Error during processing: {str(e)}")

    def updateProgress9(self, value, message):
        """更新进度辅助函数"""
        self.ui.progressBar9.setValue(value)
        self.ui.progressBar9.setFormat(message)
        slicer.app.processEvents()


    # 批量头骨剥离
    def onDialogShow6(self):
        # 显示对话框
        self.ui.dialog6.exec_()

    def onPath6_1Edited(self):
        """当第一个路径编辑框失去焦点时调用"""
        path = self.ui.lineEdit_6_1.text
        if not path:
            return
        
        if not os.path.isdir(path):
            slicer.util.errorDisplay(f"Path does not exist or is not a directory: {path}")
            return
        
        # 获取子文件夹
        try:
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            subdirs.sort()
            
            # 在显示框中显示子文件夹
            self.ui.textEdit_display6.clear()
            if subdirs:
                self.ui.textEdit_display6.append("Subdirectories:")
                for subdir in subdirs:
                    self.ui.textEdit_display6.append(f" - {subdir}")
            else:
                self.ui.textEdit_display6.append("No subdirectories found")
        except Exception as e:
            slicer.util.errorDisplay(f"Error reading directory: {str(e)}")

    def onCancel6(self):
        """取消按钮点击事件"""
        self.ui.dialog6.close()

    def onApplyClicked6(self):
        """Apply按钮点击事件"""
        # 获取输入参数
        base_dir = self.ui.lineEdit_6_1.text.strip()
        filename = self.ui.lineEdit_6_2.text.strip()
        output_name = self.ui.lineEdit_6_3.text.strip()
        output_label_name = self.ui.lineEdit_6_4.text.strip()
        
        # 验证输入
        if not all([base_dir, filename, output_label_name, output_name]):
            slicer.util.errorDisplay("All fields are required!")
            return
        
        if not os.path.isdir(base_dir):
            slicer.util.errorDisplay(f"Base directory does not exist: {base_dir}")
            return
        
        # 开始处理        
        try:
            # 获取所有子文件夹名
            subdirs = [d for d in os.listdir(base_dir) 
                    if os.path.isdir(os.path.join(base_dir, d))]
            subdirs.sort()
            
            if not subdirs:
                slicer.util.errorDisplay("no subdirs", windowTitle="Error dir path")
                return
            
            # 初始化
            self.ui.progressBar6.setValue(0)
            self.ui.progressBar6.setFormat("Prepare skull strip...")
            
            
            # 遍历每个子文件夹
            for i, subdir in enumerate(subdirs):
                self.updateProgress6(10 + i*80, f"Processing {i}/{len(subdirs)} {subdir}...")
                subdir_path = os.path.join(base_dir, subdir)
                print(f"Processing subdir: {subdir_path}")
                
                # 查找指定文件
                file_path = os.path.join(subdir_path, filename)
                output_file_path = os.path.join(subdir_path, output_name)
                output_mask_file_path = os.path.join(subdir_path, output_label_name)
                if os.path.isfile(file_path):
                    self.ui.progressBar6.setMaximum(10 + len(subdirs)*80)  # 分配权重

                    self.image_skull = slicer.util.loadVolume(file_path)

                    # 创建临时节点用于处理
                    temp_stripped_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "temp_stripped")
                    temp_mask_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "temp_mask")
                    
                    try:
                        # 运行 SwissSkullStripper
                        parameters = {
                            "patientVolume": self.image_skull.GetID(),
                            "patientOutputVolume": temp_stripped_node.GetID(),
                            "patientMaskLabel": temp_mask_node.GetID(),
                        }
                        print("正在运行头骨剥离...")
                        cli_node = slicer.cli.run(slicer.modules.swissskullstripper, None, parameters, wait_for_completion=True)

                        # 检查是否成功完成
                        if cli_node.GetStatus() != cli_node.Completed:
                            raise RuntimeError(f"头骨剥离失败，状态: {cli_node.GetStatusString()}")
                        
                        print("头骨剥离完成，正在保存结果...")
                        
                        # 保存结果到指定路径
                        print(f"保存头骨剥离结果到: {output_file_path}")
                        slicer.util.saveNode(temp_stripped_node, output_file_path)
                        
                        print(f"保存掩码结果到: {output_mask_file_path}")
                        slicer.util.saveNode(temp_mask_node, output_mask_file_path)
                        
                        print("头骨剥离完成!")
                        
                    except Exception as e:
                        raise RuntimeError(f"头骨剥离过程失败: {str(e)}")
                    
                    finally:
                        # 清理临时节点
                        slicer.mrmlScene.RemoveNode(temp_stripped_node)
                        slicer.mrmlScene.RemoveNode(temp_mask_node)
                        
                        # 可选：也移除输入节点以保持场景清洁
                        if hasattr(self, 'image_skull') and self.image_skull:
                            slicer.mrmlScene.RemoveNode(self.image_skull)

                    # # 构建 mri_synthstrip 命令
                    # mri_synthstrip_script = os.path.join(os.path.dirname(__file__), "./tools/mri_synthstrip.py")

                    # if not os.path.exists(mri_synthstrip_script):
                    #     raise FileNotFoundError(f"mri_synthstrip.py not found at: {mri_synthstrip_script}")
                    
                    # # 设置命令参数
                    # import subprocess
                    # cmd = [
                    #     "python", mri_synthstrip_script,
                    #     "-i", file_path,
                    #     "-o", output_file_path,
                    #     "-m", output_mask_file_path,
                    #     "--model", "E:\\synthstrip.1.pt"
                    # ]

                    # print(f"Running command: {' '.join(cmd)}")
                    # result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))

                    # if result.returncode != 0:
                    #     raise RuntimeError(f"mri_synthstrip failed: {result.stderr}")
                    
                    # print(f"mri_synthstrip output: {result.stdout}")
                    

            self.updateProgress6(10 + len(subdirs)*80, "Complete skull strip!")  
            
        except Exception as e:
            slicer.util.errorDisplay(f"Error during processing: {str(e)}")
  
    def updateProgress6(self, value, message):
        """更新进度辅助函数"""
        self.ui.progressBar6.setValue(value)
        self.ui.progressBar6.setFormat(message)
        slicer.app.processEvents()



    def onDialogShow6_pet(self):
        # 显示对话框
        self.ui.dialog6_pet.exec_()

    def onPath6_1Edited_pet(self):
        """当第一个路径编辑框失去焦点时调用"""
        path = self.ui.lineEdit_6_1_pet.text
        if not path:
            return
        
        if not os.path.isdir(path):
            slicer.util.errorDisplay(f"Path does not exist or is not a directory: {path}")
            return
        
        # 获取子文件夹
        try:
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            subdirs.sort()
            
            # 在显示框中显示子文件夹
            self.ui.textEdit_display6_pet.clear()
            if subdirs:
                self.ui.textEdit_display6_pet.append("Subdirectories:")
                for subdir in subdirs:
                    self.ui.textEdit_display6_pet.append(f" - {subdir}")
            else:
                self.ui.textEdit_display6_pet.append("No subdirectories found")
        except Exception as e:
            slicer.util.errorDisplay(f"Error reading directory: {str(e)}")

    def onCancel6_pet(self):
        """取消按钮点击事件"""
        self.ui.dialog6_pet.close()

    def onApplyClicked6_pet(self):
        """Apply按钮点击事件"""
        # 获取输入参数
        base_dir = self.ui.lineEdit_6_1_pet.text.strip()
        filename = self.ui.lineEdit_6_2_pet.text.strip()
        mask_name = self.ui.lineEdit_6_3_pet.text.strip()
        output_image_name = self.ui.lineEdit_6_4_pet.text.strip()
        
        # 验证输入
        if not all([base_dir, filename, mask_name, output_image_name]):
            slicer.util.errorDisplay("All fields are required!")
            return
        
        if not os.path.isdir(base_dir):
            slicer.util.errorDisplay(f"Base directory does not exist: {base_dir}")
            return
        
        # 开始处理        
        try:
            # 获取所有子文件夹名
            subdirs = [d for d in os.listdir(base_dir) 
                    if os.path.isdir(os.path.join(base_dir, d))]
            subdirs.sort()
            
            if not subdirs:
                slicer.util.errorDisplay("no subdirs", windowTitle="Error dir path")
                return
            
            # 初始化
            self.ui.progressBar6_pet.setValue(0)
            self.ui.progressBar6_pet.setFormat("Prepare skull strip...")
   
            
            # 遍历每个子文件夹
            for i, subdir in enumerate(subdirs):
                self.updateProgress6_pet(int((i / len(subdirs)) * 90), f"Processing {i}/{len(subdirs)} {subdir}...")
                subdir_path = os.path.join(base_dir, subdir)
                print(f"Processing subdir: {subdir_path}")
                
                # 查找指定文件
                file_path = os.path.join(subdir_path, filename)
                mask_path = os.path.join(subdir_path, mask_name)
                output_image_path = os.path.join(subdir_path, output_image_name)

                # 检查文件是否存在
                if not os.path.isfile(file_path):
                    print(f"File not found: {file_path}")
                    continue
                    
                if not os.path.isfile(mask_path):
                    print(f"Mask file not found: {mask_path}")
                    continue

                # 使用nibabel加载图像和掩码
                try:
                    # 加载图像
                    img = nib.load(file_path)
                    img_data = img.get_fdata()
                    
                    # 加载掩码
                    mask = nib.load(mask_path)
                    mask_data = mask.get_fdata()
                    
                    # 确保图像和掩码尺寸一致
                    if img_data.shape != mask_data.shape:
                        print(f"Image and mask shape mismatch in {subdir}: {img_data.shape} vs {mask_data.shape}")
                        continue
                    
                    # 应用掩码进行颅骨剥离（假设掩码是二值的，1为保留区域，0为背景）
                    skull_stripped_data = img_data * mask_data
                    
                    # 创建新的NIfTI图像
                    skull_stripped_img = nib.Nifti1Image(skull_stripped_data, img.affine, img.header)
                    
                    # 保存结果
                    nib.save(skull_stripped_img, output_image_path)
                    print(f"Skull-stripped image saved to: {output_image_path}")
                    
                except Exception as e:
                    print(f"Error processing {subdir}: {str(e)}")
                    continue
                

            self.updateProgress6_pet(100, "Complete skull strip!")  
            
        except Exception as e:
            slicer.util.errorDisplay(f"Error during processing: {str(e)}")

    def updateProgress6_pet(self, value, message):
        """更新进度辅助函数"""
        self.ui.progressBar6_pet.setValue(value)
        self.ui.progressBar6_pet.setFormat(message)
        slicer.app.processEvents()

    # 批量CTClip
    def onDialogShow7(self):
        # 显示对话框
        self.ui.dialog7.exec_()

    def onPath7_1Edited(self):
        """当第一个路径编辑框失去焦点时调用"""
        path = self.ui.lineEdit_7_1.text
        if not path:
            return
        
        if not os.path.isdir(path):
            slicer.util.errorDisplay(f"Path does not exist or is not a directory: {path}")
            return
        
        # 获取子文件夹
        try:
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            subdirs.sort()
            
            # 在显示框中显示子文件夹
            self.ui.textEdit_display7.clear()
            if subdirs:
                self.ui.textEdit_display7.append("Subdirectories:")
                for subdir in subdirs:
                    self.ui.textEdit_display7.append(f" - {subdir}")
            else:
                self.ui.textEdit_display7.append("No subdirectories found")
        except Exception as e:
            slicer.util.errorDisplay(f"Error reading directory: {str(e)}")

    def onCancel7(self):
        """取消按钮点击事件"""
        self.ui.dialog7.close()

    def onApplyClicked7(self):
        """Apply按钮点击事件"""
        # 获取输入参数
        base_dir = self.ui.lineEdit_7_1.text.strip()
        filename = self.ui.lineEdit_7_2.text.strip()
        minimum_str = self.ui.lineEdit_7_3.text.strip()
        maximum_str = self.ui.lineEdit_7_4.text.strip()
        normalize = self.ui.CTClipNormalizeComboBox_batch.currentText
        output_name = self.ui.lineEdit_7_5.text.strip()
        
        # 验证输入
        if not all([base_dir, filename, minimum_str, maximum_str, output_name]):
            slicer.util.errorDisplay("All fields are required!")
            return
        
        if not os.path.isdir(base_dir):
            slicer.util.errorDisplay(f"Base directory does not exist: {base_dir}")
            return
        
        # 转换为数字
        try:
            minimum = float(minimum_str)
        except ValueError:
            slicer.util.errorDisplay(f"Invalid minimum value: {minimum_str}")
            return
            
        try:
            maximum = float(maximum_str)
        except ValueError:
            slicer.util.errorDisplay(f"Invalid maximum value: {maximum_str}")
            return
        
        # 开始处理        
        try:
            # 获取所有子文件夹名
            subdirs = [d for d in os.listdir(base_dir) 
                    if os.path.isdir(os.path.join(base_dir, d))]
            subdirs.sort()
            
            if not subdirs:
                slicer.util.errorDisplay("no subdirs", windowTitle="Error dir path")
                return
            
            # 初始化
            self.ui.progressBar7.setValue(0)
            self.ui.progressBar7.setFormat("Prepare CT Clip...")
            
            
            # 遍历每个子文件夹
            for i, subdir in enumerate(subdirs):
                self.updateProgress7(10 + i*80, f"Processing {i}/{len(subdirs)} {subdir}...")
                subdir_path = os.path.join(base_dir, subdir)
                print(f"Processing subdir: {subdir_path}")
                
                # 查找指定文件
                file_path = os.path.join(subdir_path, filename)
                output_file_path = os.path.join(subdir_path, output_name)

                if os.path.isfile(file_path):
                    self.ui.progressBar7.setMaximum(10 + len(subdirs)*80)  # 分配权重

                    data, affine, header = self.logic.load_nifti(file_path)
                    normalized_data = self.logic.threshold_and_normalize(data, minimum, maximum, normalize)

                    self.logic.save_nifti(normalized_data, affine, header, output_file_path)
                    print(f"CTClip image saved to: {output_file_path}")
                    
            self.updateProgress7(10 + len(subdirs)*80, "Complete CT Clip!")  
            
        except Exception as e:
            slicer.util.errorDisplay(f"Error during processing: {str(e)}")
  
    def updateProgress7(self, value, message):
        """更新进度辅助函数"""
        self.ui.progressBar7.setValue(value)
        self.ui.progressBar7.setFormat(message)
        slicer.app.processEvents()



    # 批量刚性配准
    def onDialogShow1(self):
        # 显示对话框
        self.ui.dialog1.exec_()

    def onPath1_1Edited(self):
        """当第一个路径编辑框失去焦点时调用"""
        path = self.ui.lineEdit_1_1.text
        if not path:
            return
        
        if not os.path.isdir(path):
            slicer.util.errorDisplay(f"Path does not exist or is not a directory: {path}")
            return
        
        # 获取子文件夹
        try:
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            subdirs.sort()
            
            # 在显示框中显示子文件夹
            self.ui.textEdit_display1.clear()
            if subdirs:
                self.ui.textEdit_display1.append("Subdirectories:")
                for subdir in subdirs:
                    self.ui.textEdit_display1.append(f" - {subdir}")
            else:
                self.ui.textEdit_display1.append("No subdirectories found")
        except Exception as e:
            slicer.util.errorDisplay(f"Error reading directory: {str(e)}")

    def onCancel1(self):
        """取消按钮点击事件"""
        self.ui.dialog1.close()

    def onApplyClicked1(self):
        """Apply按钮点击事件"""
        # 获取输入参数
        base_dir = self.ui.lineEdit_1_1.text.strip()
        filename = self.ui.lineEdit_1_2.text.strip()
        fixed_image_path = self.ui.lineEdit_1_3.text.strip()
        output_name = self.ui.lineEdit_1_4.text.strip()
        output_field_name = self.ui.lineEdit_1_5.text.strip()
        interpolation_mode = self.ui.interpolationComboBox_rigid_batch.currentText

        print(f"base_dir: {base_dir}")
        print(f"filename: {filename}")
        print(f"fixed_image_path: {fixed_image_path}")
        print(f'interpolation_mode: {interpolation_mode}')
        print(f"output_name: {output_name}")
        print(f"output_field_name: {output_name}")
        
        # 验证输入
        if not all([base_dir, filename, fixed_image_path, output_name, output_field_name]):
            slicer.util.errorDisplay("All fields are required!")
            return
        
        if not os.path.isdir(base_dir):
            slicer.util.errorDisplay(f"Base directory does not exist: {base_dir}")
            return
        
        if not os.path.isfile(fixed_image_path):
            slicer.util.errorDisplay(f"Fixed image file does not exist: {fixed_image_path}")
            return
        
        # 开始处理        
        try:
            # 获取所有子文件夹名
            subdirs = [d for d in os.listdir(base_dir) 
                    if os.path.isdir(os.path.join(base_dir, d))]
            subdirs.sort()
            
            if not subdirs:
                slicer.util.errorDisplay("no subdirs", windowTitle="Error dir path")
                return
            
            # 初始化
            self.ui.progressBar1.setValue(0)
            self.ui.progressBar1.setFormat("Prepare register...")
            
            self.updateProgress1(10, "Load fixed image...")
            fixed_volume  = slicer.util.loadVolume(fixed_image_path)
            
            # 遍历每个子文件夹
            for i, subdir in enumerate(subdirs):
                self.updateProgress1(10 + i*80, f"Processing {i}/{len(subdirs)} {subdir}...")
                subdir_path = os.path.join(base_dir, subdir)
                print(f"Processing subdir: {subdir_path}")
                
                # 查找指定文件
                file_path = os.path.join(subdir_path, filename)
                if os.path.isfile(file_path):
                    moving_volume  = slicer.util.loadVolume(file_path)
                    self.ui.progressBar1.setMaximum(10 + len(subdirs)*80)  # 分配权重

                    # Create output volume node
                    self.rigidRegisteredVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", output_name)
                    self.rigidRegisteredFieldVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", output_field_name)

                    # 创建配准参数
                    parameters = {
                        "fixedVolume": fixed_volume.GetID(),
                        "movingVolume": moving_volume.GetID(),
                        "outputVolume": self.rigidRegisteredVolumeNode.GetID(),
                        "outputTransform": self.rigidRegisteredFieldVolumeNode.GetID(),  # 添加形变场输出参数
                        "useRigid": True,
                        "useAffine": False,
                        "samplingPercentage": 0.02,
                        "initializeTransformMode": "useGeometryAlign",
                        "interpolationMode": interpolation_mode,
                    }

                    # 执行刚性配准 (使用BRAINSFit模块)
                    registration = slicer.cli.runSync(
                        slicer.modules.brainsfit,
                        None,  # 无自定义节点
                        parameters
                    )

                    if not registration.GetStatusString() == "Completed":
                        raise RuntimeError(f"Error rigid register: {registration.GetStatusString()}")
                    
                    # 保存节点到文件
                    output_path = os.path.join(subdir_path, f"{output_name}.nii.gz")
                    output_field_path = os.path.join(subdir_path, f"{output_field_name}.h5")

                    try:
                        slicer.util.saveNode(self.rigidRegisteredVolumeNode, output_path)
                        slicer.util.saveNode(self.rigidRegisteredFieldVolumeNode, output_field_path)
                        print(f"rigid registation result is saved to: {output_path}")
                        print(f"rigid field registation result is saved to: {output_field_path}")
                    except Exception as e:
                        print(f"Error save: {str(e)}")
                        slicer.util.errorDisplay(f"Error save: {str(e)}")
                    

                    # 清理临时节点
                    slicer.mrmlScene.RemoveNode(moving_volume)
                    slicer.mrmlScene.RemoveNode(self.rigidRegisteredVolumeNode)
                    slicer.mrmlScene.RemoveNode(self.rigidRegisteredFieldVolumeNode)

            self.updateProgress1(10 + len(subdirs)*80, "Complete registration!")  
            
        except Exception as e:
            slicer.util.errorDisplay(f"Error during processing: {str(e)}")

        finally:
                if 'fixed_volume' in locals() and fixed_volume:
                    slicer.mrmlScene.RemoveNode(fixed_volume)
  
    def updateProgress1(self, value, message):
        """更新进度辅助函数"""
        self.ui.progressBar1.setValue(value)
        self.ui.progressBar1.setFormat(message)
        slicer.app.processEvents()



    def onDialogShow1_pet(self):
        # 显示对话框
        self.ui.dialog1_pet.exec_()

    def onPath1_1Edited_pet(self):
        """当第一个路径编辑框失去焦点时调用"""
        path = self.ui.lineEdit_1_1_pet.text
        if not path:
            return
        
        if not os.path.isdir(path):
            slicer.util.errorDisplay(f"Path does not exist or is not a directory: {path}")
            return
        
        # 获取子文件夹
        try:
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            subdirs.sort()
            
            # 在显示框中显示子文件夹
            self.ui.textEdit_display1_pet.clear()
            if subdirs:
                self.ui.textEdit_display1_pet.append("Subdirectories:")
                for subdir in subdirs:
                    self.ui.textEdit_display1_pet.append(f" - {subdir}")
            else:
                self.ui.textEdit_display1_pet.append("No subdirectories found")
        except Exception as e:
            slicer.util.errorDisplay(f"Error reading directory: {str(e)}")

    def onCancel1_pet(self):
        """取消按钮点击事件"""
        self.ui.dialog1_pet.close()

    def onApplyClicked1_pet(self):
        """Apply按钮点击事件"""
        # 获取输入参数
        base_dir = self.ui.lineEdit_1_1_pet.text.strip()
        filename = self.ui.lineEdit_1_2_pet.text.strip()
        field_name = self.ui.lineEdit_1_3_pet.text.strip()
        output_name = self.ui.lineEdit_1_4_pet.text.strip()

        print(f"base_dir: {base_dir}")
        print(f"filename: {filename}")
        print(f"field_name: {field_name}")
        print(f"output_name: {output_name}")
        
        # 验证输入
        if not all([base_dir, filename, field_name, output_name]):
            slicer.util.errorDisplay("All fields are required!")
            return
        
        if not os.path.isdir(base_dir):
            slicer.util.errorDisplay(f"Base directory does not exist: {base_dir}")
            return
        
        
        # 开始处理        
        try:
            # 获取所有子文件夹名
            subdirs = [d for d in os.listdir(base_dir) 
                    if os.path.isdir(os.path.join(base_dir, d))]
            subdirs.sort()
            
            if not subdirs:
                slicer.util.errorDisplay("no subdirs", windowTitle="Error dir path")
                return
            
            # 初始化
            self.ui.progressBar1.setValue(0)
            self.ui.progressBar1.setFormat("Prepare register...")
            
            # 遍历每个子文件夹
            for i, subdir in enumerate(subdirs):
                self.updateProgress1(10 + i*90, f"Processing {i}/{len(subdirs)} {subdir}...")
                subdir_path = os.path.join(base_dir, subdir)
                print(f"Processing subdir: {subdir_path}")
                
                # 查找指定文件
                file_path = os.path.join(subdir_path, filename)
                field_path = os.path.join(subdir_path, field_name)
                if os.path.isfile(file_path):
                    moving_volume  = slicer.util.loadVolume(file_path)
                    field = slicer.util.loadTransform(field_path)
                    self.ui.progressBar1.setMaximum(10 + len(subdirs)*90)  # 分配权重

                    parameters = {
                        "inputVolume": moving_volume.GetID(),
                        "referenceVolume": moving_volume.GetID(),  # 可以用自己，或另一张图作为参考空间
                        "outputVolume": slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", output_name).GetID(),
                        "warpTransform": field.GetID()
                    }

                    cliNode = slicer.cli.runSync(slicer.modules.brainsresample, None, parameters)
                    outputNode = slicer.mrmlScene.GetNodeByID(parameters["outputVolume"])


                    if cliNode.GetStatusString() == "Completed":
                        print("Rigid registration applied successfully.")
                        # 保存节点到文件
                        output_path = os.path.join(subdir_path, f"{output_name}.nii.gz")
                        try:
                            slicer.util.saveNode(outputNode, output_path)
                            print(f"rigid registation result is saved to: {output_path}")
                        except Exception as e:
                            print(f"Error save: {str(e)}")
                            slicer.util.errorDisplay(f"Error save: {str(e)}")
                    else:
                        slicer.util.errorDisplay("Failed to apply transform. Check the log for details.")

                    # 清理临时节点
                    slicer.mrmlScene.RemoveNode(moving_volume)
                    slicer.mrmlScene.RemoveNode(field)
                    slicer.mrmlScene.RemoveNode(outputNode)

            self.updateProgress1(10 + len(subdirs)*90, "Complete registration!")  
            
        except Exception as e:
            slicer.util.errorDisplay(f"Error during processing: {str(e)}")

    def updateProgress1_pet(self, value, message):
        """更新进度辅助函数"""
        self.ui.progressBar1_pet.setValue(value)
        self.ui.progressBar1_pet.setFormat(message)
        slicer.app.processEvents()


    # 批量空间配准
    def onDialogShow2(self):
        # 显示对话框
        self.ui.dialog2.exec_()

    def onPath2_1Edited(self):
        """当第一个路径编辑框失去焦点时调用"""
        path = self.ui.lineEdit_2_1.text
        if not path:
            return
        
        if not os.path.isdir(path):
            slicer.util.errorDisplay(f"Path does not exist or is not a directory: {path}")
            return
        
        # 获取子文件夹
        try:
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            subdirs.sort()
            
            # 在显示框中显示子文件夹
            self.ui.textEdit_display2.clear()
            if subdirs:
                self.ui.textEdit_display2.append("Subdirectories:")
                for subdir in subdirs:
                    self.ui.textEdit_display2.append(f" - {subdir}")
            else:
                self.ui.textEdit_display2.append("No subdirectories found")
        except Exception as e:
            slicer.util.errorDisplay(f"Error reading directory: {str(e)}")

    def onCancel2(self):
        """取消按钮点击事件"""
        self.ui.dialog2.close()

    def onApplyClicked2(self):
        """Apply按钮点击事件"""
        # 获取输入参数
        base_dir = self.ui.lineEdit_2_1.text.strip()
        filename = self.ui.lineEdit_2_2.text.strip()
        fixed_image_path = self.ui.lineEdit_2_3.text.strip()
        resolution_text = self.ui.lineEdit_2_4.text.strip()
        dimensions_text = self.ui.lineEdit_2_5.text.strip()
        output_name = self.ui.lineEdit_2_6.text.strip()
        interpolation_mode = self.ui.interpolationComboBox_spaceRegister_batch.currentText

        # 验证输入
        if not all([base_dir, filename, fixed_image_path, output_name]):
            slicer.util.errorDisplay("All fields are required!")
            return
        
        if not os.path.isdir(base_dir):
            slicer.util.errorDisplay(f"Base directory does not exist: {base_dir}")
            return
        
        if not os.path.isfile(fixed_image_path):
            slicer.util.errorDisplay(f"Fixed image file does not exist: {fixed_image_path}")
            return

        try:
            dimensions_list = [int(d) for d in dimensions_text.split(',')]
            if len(dimensions_list) != 3:
                raise ValueError
            dimensions = tuple(dimensions_list)

            resolution_list = [int(d) for d in resolution_text.split(',')]
            if len(resolution_list) != 3:
                raise ValueError
            resolutions = tuple(resolution_list)
        except:
            slicer.util.errorDisplay("Please input the form of resolution or dimension, such as: 128,128,128")
            return
        
        out_dir = '.\\temp_data'
        # Threading.
        tf.config.threading.set_inter_op_parallelism_threads(12)
        tf.config.threading.set_intra_op_parallelism_threads(12)
        
        # 开始处理
        try:
            # 获取所有子文件夹名
            subdirs = [d for d in os.listdir(base_dir) 
                    if os.path.isdir(os.path.join(base_dir, d))]
            subdirs.sort()
            
            if not subdirs:
                slicer.util.errorDisplay("no subdirs", windowTitle="Error dir path")
                return
            
            # 初始化
            self.ui.progressBar2.setValue(0)
            self.ui.progressBar2.setFormat("Prepare register...")
            
            self.updateProgress(10, "Load fixed image...")
            fix = sf.load_volume(fixed_image_path)
            
            # 遍历每个子文件夹
            for i, subdir in enumerate(subdirs):
                self.updateProgress(10 + i*80, f"Processing {i}/{len(subdirs)} {subdir}...")
                subdir_path = os.path.join(base_dir, subdir)
                print(f"Processing subdir: {subdir_path}")
                
                # 查找指定文件
                file_path = os.path.join(subdir_path, filename)
                if os.path.isfile(file_path):
                    # Input data.
                    mov = sf.load_volume(file_path)
                    self.ui.progressBar2.setMaximum(10 + len(subdirs)*80)  # 分配权重
                    
                    if not len(mov.shape) == len(fix.shape) == 3:
                        sf.system.fatal('input images are not single-frame volumes')

                    center = fix
                    net_to_mov, mov_to_net = self.logic.network_space(im=mov, shape=dimensions, voxsize=resolutions, center=center)
                    net_to_fix, fix_to_net = self.logic.network_space(fix, shape=dimensions, voxsize=resolutions)

                    mov_to_ras = mov.geom.vox2world.matrix
                    fix_to_ras = fix.geom.vox2world.matrix

                    inputs = (
                            self.logic.transform(mov, net_to_mov, shape=dimensions, normalize=False, batch=True, interpolationComboBox_mode=interpolation_mode),
                            self.transform(fix, net_to_fix, shape=dimensions, normalize=False, batch=True, interpolationComboBox_mode=interpolation_mode),
                        )

                    os.makedirs(out_dir, exist_ok=True)

                    inp_1 = os.path.join(out_dir, f'{output_name}.nii')
                    inp_2 = os.path.join(out_dir, 'fixed.nii')
                    geom_1 = sf.ImageGeometry(dimensions, vox2world=mov_to_ras @ net_to_mov)
                    geom_2 = sf.ImageGeometry(dimensions, vox2world=fix_to_ras @ net_to_fix)
                    sf.Volume(inputs[0][0], geom_1).save(inp_1)
                    sf.Volume(inputs[1][0], geom_2).save(inp_2)

                    # Load the saved volume into 3D Slicer
                    loadedVolumeNode = slicer.util.loadVolume(inp_1)
                    
                    # Optionally set the name of the loaded volume
                    loadedVolumeNode.SetName(output_name)

                    # 保存节点到文件
                    output_path = os.path.join(subdir_path, f"{output_name}.nii.gz")
                    try:
                        slicer.util.saveNode(loadedVolumeNode, output_path)
                        print(f"space registation result is saved to: {output_path}")
                    except Exception as e:
                        print(f"Error save: {str(e)}")
                        slicer.util.errorDisplay(f"Error save: {str(e)}")
                    

                    # 清理临时节点
                    slicer.mrmlScene.RemoveNode(loadedVolumeNode)

            self.updateProgress(10 + len(subdirs)*80, "Complete registration!")  
            
        except Exception as e:
            slicer.util.errorDisplay(f"Error during processing: {str(e)}")

    def updateProgress(self, value, message):
        """更新进度辅助函数"""
        self.ui.progressBar2.setValue(value)
        self.ui.progressBar2.setFormat(message)
        slicer.app.processEvents()


    # 批量synthmorph配准
    def onDialogShow3(self):
        # 显示对话框
        self.ui.dialog3.exec_()

    def onPath3_1Edited(self):
        """当第一个路径编辑框失去焦点时调用"""
        path = self.ui.lineEdit_3_1.text
        if not path:
            return
        
        if not os.path.isdir(path):
            slicer.util.errorDisplay(f"Path does not exist or is not a directory: {path}")
            return
        
        # 获取子文件夹
        try:
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            subdirs.sort()
            
            # 在显示框中显示子文件夹
            self.ui.textEdit_display3.clear()
            if subdirs:
                self.ui.textEdit_display3.append("Subdirectories:")
                for subdir in subdirs:
                    self.ui.textEdit_display3.append(f" - {subdir}")
            else:
                self.ui.textEdit_display3.append("No subdirectories found")
        except Exception as e:
            slicer.util.errorDisplay(f"Error reading directory: {str(e)}")

    def onCancel3(self):
        """取消按钮点击事件"""
        self.ui.dialog3.close()

    def onApplyClicked3(self):
        """Apply按钮点击事件"""
        # 获取输入参数
        base_dir = self.ui.lineEdit_3_1.text.strip()
        filename = self.ui.lineEdit_3_2.text.strip()
        fixed_image_path = self.ui.lineEdit_3_3.text.strip()
        output_name = self.ui.lineEdit_3_4.text.strip()
        output_field_name = self.ui.lineEdit_3_5.text.strip()

        # 验证输入
        if not all([base_dir, filename, fixed_image_path, output_name, output_field_name]):
            slicer.util.errorDisplay("All fields are required!")
            return
        
        if not os.path.isdir(base_dir):
            slicer.util.errorDisplay(f"Base directory does not exist: {base_dir}")
            return
        
        if not os.path.isfile(fixed_image_path):
            slicer.util.errorDisplay(f"Fixed image file does not exist: {fixed_image_path}")
            return
        
        import subprocess

        # 开始处理        
        try:
            # 获取所有子文件夹名
            subdirs = [d for d in os.listdir(base_dir) 
                    if os.path.isdir(os.path.join(base_dir, d))]
            subdirs.sort()
            
            if not subdirs:
                slicer.util.errorDisplay("no subdirs", windowTitle="Error dir path")
                return
            
            # 初始化
            self.ui.progressBar3.setValue(0)
            self.ui.progressBar3.setFormat("Prepare register...")
            
            # 遍历每个子文件夹
            for i, subdir in enumerate(subdirs):
                self.updateProgress3(10 + i*80, f"Processing {i}/{len(subdirs)} {subdir}...")
                subdir_path = os.path.join(base_dir, subdir)
                print(f"Processing subdir: {subdir_path}")
                
                # 查找指定文件
                file_path = os.path.join(subdir_path, filename)
                if os.path.isfile(file_path):
                    self.ui.progressBar3.setMaximum(10 + len(subdirs)*80)  # 分配权重

                    output_path = os.path.join(subdir_path, f"{output_name}.nii.gz")
                    output_field_path = os.path.join(subdir_path, f"{output_field_name}.nii.gz")

                    cmd = [
                        r".\bin\PythonSlicer.exe",  # 确保使用 Slicer 的 Python
                        r".\slicer.org\Extensions-33241\PETCTREGISTER\PETCTREGISTER\mri_synthmorph\mri_synthmorph.py", 
                        "register",
                        file_path,
                        fixed_image_path,
                        "-m", "joint",
                        "-o", output_path,
                        "-t", output_field_path
                    ]

                    try:
                        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='latin-1')
                        print(result.stdout)
                    except subprocess.CalledProcessError as e:
                        print("Command failed with error:")
                        print(e.stderr)

            self.updateProgress3(10 + len(subdirs)*80, "Complete registration!")  
            
        except Exception as e:
            slicer.util.errorDisplay(f"Error during processing: {str(e)}")

    def updateProgress3(self, value, message):
        """更新进度辅助函数"""
        self.ui.progressBar3.setValue(value)
        self.ui.progressBar3.setFormat(message)
        slicer.app.processEvents()



    def onDialogShow3_pet(self):
        # 显示对话框
        self.ui.dialog3_pet.exec_()

    def onPath3_1Edited_pet(self):
        """当第一个路径编辑框失去焦点时调用"""
        path = self.ui.lineEdit_3_1_pet.text
        if not path:
            return
        
        if not os.path.isdir(path):
            slicer.util.errorDisplay(f"Path does not exist or is not a directory: {path}")
            return
        
        # 获取子文件夹
        try:
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            subdirs.sort()
            
            # 在显示框中显示子文件夹
            self.ui.textEdit_display3_pet.clear()
            if subdirs:
                self.ui.textEdit_display3_pet.append("Subdirectories:")
                for subdir in subdirs:
                    self.ui.textEdit_display3_pet.append(f" - {subdir}")
            else:
                self.ui.textEdit_display3_pet.append("No subdirectories found")
        except Exception as e:
            slicer.util.errorDisplay(f"Error reading directory: {str(e)}")

    def onCancel3_pet(self):
        """取消按钮点击事件"""
        self.ui.dialog3_pet.close()

    def onApplyClicked3_pet(self):
        """Apply按钮点击事件"""
        # 获取输入参数
        base_dir = self.ui.lineEdit_3_1_pet.text.strip()
        filename = self.ui.lineEdit_3_2_pet.text.strip()
        field_name = self.ui.lineEdit_3_3_pet.text.strip()
        output_name = self.ui.lineEdit_3_4_pet.text.strip()
        interpolation_mode = self.ui.interpolationComboBox_synthmorph_apply_batch.currentText

        # 验证输入
        if not all([base_dir, filename, field_name, output_name]):
            slicer.util.errorDisplay("All fields are required!")
            return
        
        if not os.path.isdir(base_dir):
            slicer.util.errorDisplay(f"Base directory does not exist: {base_dir}")
            return
        
        import subprocess

        # 开始处理        
        try:
            # 获取所有子文件夹名
            subdirs = [d for d in os.listdir(base_dir) 
                    if os.path.isdir(os.path.join(base_dir, d))]
            subdirs.sort()
            
            if not subdirs:
                slicer.util.errorDisplay("no subdirs", windowTitle="Error dir path")
                return
            
            # 初始化
            self.ui.progressBar3_pet.setValue(0)
            self.ui.progressBar3_pet.setFormat("Prepare register...")
            
            # 遍历每个子文件夹
            for i, subdir in enumerate(subdirs):
                self.updateProgress3_pet(10 + i*80, f"Processing {i}/{len(subdirs)} {subdir}...")
                subdir_path = os.path.join(base_dir, subdir)
                print(f"Processing subdir: {subdir_path}")
                
                # 查找指定文件
                file_path = os.path.join(subdir_path, filename)
                if os.path.isfile(file_path):
                    self.ui.progressBar3_pet.setMaximum(10 + len(subdirs)*80)  # 分配权重

                    output_path = os.path.join(subdir_path, f"{output_name}.nii.gz")
                    field_path = os.path.join(subdir_path, f"{field_name}.nii.gz")
                    filename_path = os.path.join(subdir_path, filename)

                    cmd = [
                        r".\bin\PythonSlicer.exe",  # 确保使用 Slicer 的 Python
                        r".\slicer.org\Extensions-33241\PETCTREGISTER\PETCTREGISTER\mri_synthmorph\mri_synthmorph.py", 
                        "apply",
                        field_path,
                        filename_path,
                        output_path,
                        "-m", interpolation_mode
                    ]

                    try:
                        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='latin-1')
                        print(result.stdout)
                    except subprocess.CalledProcessError as e:
                        print("Command failed with error:")
                        print(e.stderr)

            self.updateProgress3_pet(10 + len(subdirs)*80, "Complete registration!")  
            
        except Exception as e:
            slicer.util.errorDisplay(f"Error during processing: {str(e)}")

    def updateProgress3_pet(self, value, message):
        """更新进度辅助函数"""
        self.ui.progressBar3_pet.setValue(value)
        self.ui.progressBar3_pet.setFormat(message)
        slicer.app.processEvents()


    # 批量Dice计算
    def onDialogShow4(self):
        # 显示对话框
        self.ui.dialog4.exec_()

    def onPath4_1Edited(self):
        """当第一个路径编辑框失去焦点时调用"""
        path = self.ui.lineEdit_4_1.text
        if not path:
            return
        
        if not os.path.isdir(path):
            slicer.util.errorDisplay(f"Path does not exist or is not a directory: {path}")
            return
        
        # 获取子文件夹
        try:
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            subdirs.sort()
            
            # 在显示框中显示子文件夹
            self.ui.textEdit_display4.clear()
            if subdirs:
                self.ui.textEdit_display4.append("Subdirectories:")
                for subdir in subdirs:
                    self.ui.textEdit_display4.append(f" - {subdir}")
            else:
                self.ui.textEdit_display4.append("No subdirectories found")
        except Exception as e:
            slicer.util.errorDisplay(f"Error reading directory: {str(e)}")

    def onCancel4(self):
        """取消按钮点击事件"""
        self.ui.dialog4.close()

    def onApplyClicked4(self):
        """Apply按钮点击事件"""
        # 获取输入参数
        base_dir = self.ui.lineEdit_4_1.text.strip()
        label1_name = self.ui.lineEdit_4_2.text.strip()
        label2_name = self.ui.lineEdit_4_3.text.strip()
        label_maps_text = self.ui.lineEdit_4_4.text.strip()
        output_path = self.ui.lineEdit_4_5.text.strip()

        
        # 验证输入
        if not all([base_dir, label1_name, label2_name, label_maps_text, output_path]):
            slicer.util.errorDisplay("All fields are required!")
            return
        
        if not os.path.isdir(base_dir):
            slicer.util.errorDisplay(f"Base directory does not exist: {base_dir}")
            return
        
        try:
            label_maps_list = [int(d) for d in label_maps_text.split(',')]
            label_maps = tuple(label_maps_list)
        except:
            slicer.util.errorDisplay("Please input the form of label map, such as: 128,128,128")
            return
        
        import dice_calculate as dc
        import pandas as pd
        from openpyxl import Workbook

        # 准备结果数据结构
        results = {
            'Folder': [],
            'Label_Maps': [],
            'Dice_Mean': [],
            'Dice_Std': []
        }
        
        # 开始处理        
        try:
            # 获取所有子文件夹名
            subdirs = [d for d in os.listdir(base_dir) 
                    if os.path.isdir(os.path.join(base_dir, d))]
            subdirs.sort()
            
            if not subdirs:
                slicer.util.errorDisplay("no subdirs", windowTitle="Error dir path")
                return
            
            # 初始化
            self.ui.progressBar4.setValue(0)
            self.ui.progressBar4.setFormat("Prepare register...")
            
            # 遍历每个子文件夹
            for i, subdir in enumerate(subdirs):
                self.updateProgress4(10 + i*80, f"Processing {i}/{len(subdirs)} {subdir}...")
                subdir_path = os.path.join(base_dir, subdir)
                print(f"Processing subdir: {subdir_path}")
                
                # 查找指定文件
                label1_path = os.path.join(subdir_path, label1_name)
                label2_path = os.path.join(subdir_path, label2_name)
                if os.path.isfile(label1_path) and os.path.isfile(label2_path):
                    self.ui.progressBar4.setMaximum(10 + len(subdirs)*80)  # 分配权重


                    dice_result = dc.dice_compute(label1_path, label2_path, labels=label_maps)
                    print('Dice: %.4f +/- %.4f' % (dice_result[0], dice_result[1]))

                    results['Folder'].append(subdir)
                    results['Label_Maps'].append(str(label_maps))
                    results['Dice_Mean'].append(float(dice_result[0]))
                    results['Dice_Std'].append(float(dice_result[1]))

            # 保存结果到Excel文件
            df = pd.DataFrame(results)

            # 使用openpyxl引擎以支持格式设置
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Dice_Results')
                
                # 获取工作簿和工作表对象进行格式设置
                workbook = writer.book
                worksheet = writer.sheets['Dice_Results']
                
                # 设置列宽
                worksheet.column_dimensions['A'].width = 20
                worksheet.column_dimensions['B'].width = 20
                worksheet.column_dimensions['C'].width = 15
                worksheet.column_dimensions['D'].width = 12
                worksheet.column_dimensions['E'].width = 12
                

            print(f"结果已保存到: {output_path}")
            self.updateProgress4(100, "Dice计算完成!")
            slicer.util.infoDisplay(f"Dice计算结果已保存到:\n{output_path}")

            self.updateProgress4(10 + len(subdirs)*80, "Complete registration!")  
            
        except Exception as e:
            slicer.util.errorDisplay(f"Error during processing: {str(e)}")

    def updateProgress4(self, value, message):
        """更新进度辅助函数"""
        self.ui.progressBar4.setValue(value)
        self.ui.progressBar4.setFormat(message)
        slicer.app.processEvents()

    # 批量SUVR mapping
    def onDialogShow10_pet(self):
        # 显示对话框
        self.ui.dialog10_pet.exec_()

    def onPath10_1Edited_pet(self):
        """当第一个路径编辑框失去焦点时调用"""
        path = self.ui.lineEdit_10_1_pet.text
        if not path:
            return
        
        if not os.path.isdir(path):
            slicer.util.errorDisplay(f"Path does not exist or is not a directory: {path}")
            return
        
        # 获取子文件夹
        try:
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            subdirs.sort()
            
            # 在显示框中显示子文件夹
            self.ui.textEdit_display10_pet.clear()
            if subdirs:
                self.ui.textEdit_display10_pet.append("Subdirectories:")
                for subdir in subdirs:
                    self.ui.textEdit_display10_pet.append(f" - {subdir}")
            else:
                self.ui.textEdit_display10_pet.append("No subdirectories found")
        except Exception as e:
            slicer.util.errorDisplay(f"Error reading directory: {str(e)}")

    def onCancel10_pet(self):
        """取消按钮点击事件"""
        self.ui.dialog10_pet.close()

    def onApplyClicked10_pet(self):
        """Apply按钮点击事件"""
        # 获取输入参数
        base_dir = self.ui.lineEdit_10_1_pet.text.strip()
        filename = self.ui.lineEdit_10_2_pet.text.strip()
        mask_name = self.ui.lineEdit_10_3_pet.text.strip()
        output_image_name = self.ui.lineEdit_10_4_pet.text.strip()
        
        # 验证输入
        if not all([base_dir, filename, mask_name, output_image_name]):
            slicer.util.errorDisplay("All fields are required!")
            return
        
        if not os.path.isdir(base_dir):
            slicer.util.errorDisplay(f"Base directory does not exist: {base_dir}")
            return
        
        # 开始处理        
        try:
            # 获取所有子文件夹名
            subdirs = [d for d in os.listdir(base_dir) 
                    if os.path.isdir(os.path.join(base_dir, d))]
            subdirs.sort()
            
            if not subdirs:
                slicer.util.errorDisplay("no subdirs", windowTitle="Error dir path")
                return
            
            # 初始化
            self.ui.progressBar6_pet.setValue(0)
            self.ui.progressBar6_pet.setFormat("Prepare skull strip...")
   
            
            # 遍历每个子文件夹
            for i, subdir in enumerate(subdirs):
                self.updateProgress10_pet(int((i / len(subdirs)) * 90), f"Processing {i}/{len(subdirs)} {subdir}...")
                subdir_path = os.path.join(base_dir, subdir)
                print(f"Processing subdir: {subdir_path}")
                
                # 查找指定文件
                file_path = os.path.join(subdir_path, filename)
                mask_path = os.path.join(subdir_path, mask_name)
                output_image_path = os.path.join(subdir_path, output_image_name)

                # 检查文件是否存在
                if not os.path.isfile(file_path):
                    print(f"File not found: {file_path}")
                    continue
                    
                if not os.path.isfile(mask_path):
                    print(f"Mask file not found: {mask_path}")
                    continue

                # 初始化归一化器
                normalizer = PETNormalizerWithRegistration()
                
                # 加载图像（请替换为实际路径）
                pet_path = file_path
                ref_mask_path = mask_path  # 0为背景，1为小脑灰质
                
                if not normalizer.load_images(pet_path, ref_mask_path):
                    return
                
                # 检查图像兼容性
                if not normalizer.check_image_compatibility():
                    print("图像不兼容，进行配准...")
                    # 进行配准（使用您提供的MATLAB代码方法）
                    if not normalizer.register_mask_to_pet():
                        print("配准失败，使用原始mask")
                else:
                    print("图像兼容，跳过配准步骤")
                
                # 计算SUVR
                try:
                    suvr_img, suvr_data = normalizer.calculate_suvr(use_registered_mask=True)
                    
                    # 保存结果
                    normalizer.save_suvr_image(suvr_img, output_image_path)
                    
                except Exception as e:
                    print(f"SUVR计算错误: {e}")
                

            self.updateProgress10_pet(100, "Complete skull strip!")  
            
        except Exception as e:
            slicer.util.errorDisplay(f"Error during processing: {str(e)}")

    def updateProgress10_pet(self, value, message):
        """更新进度辅助函数"""
        self.ui.progressBar10_pet.setValue(value)
        self.ui.progressBar10_pet.setFormat(message)
        slicer.app.processEvents()


    # 批量Suvr计算
    def onDialogShow5(self):
        # 显示对话框
        self.ui.dialog5.exec_()

    def onPath5_1Edited(self):
        """当第一个路径编辑框失去焦点时调用"""
        path = self.ui.lineEdit_5_1.text
        if not path:
            return
        
        if not os.path.isdir(path):
            slicer.util.errorDisplay(f"Path does not exist or is not a directory: {path}")
            return
        
        # 获取子文件夹
        try:
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            subdirs.sort()
            
            # 在显示框中显示子文件夹
            self.ui.textEdit_display5.clear()
            if subdirs:
                self.ui.textEdit_display5.append("Subdirectories:")
                for subdir in subdirs:
                    self.ui.textEdit_display5.append(f" - {subdir}")
            else:
                self.ui.textEdit_display5.append("No subdirectories found")
        except Exception as e:
            slicer.util.errorDisplay(f"Error reading directory: {str(e)}")

    def onCancel5(self):
        """取消按钮点击事件"""
        self.ui.dialog5.close()

    def onApplyClicked5(self):
        """Apply按钮点击事件"""
        # 获取输入参数
        base_dir = self.ui.lineEdit_5_1.text.strip()  # 添加括号
        pet_name = self.ui.lineEdit_5_2.text.strip()
        label_path = self.ui.lineEdit_5_3.text.strip()
        label_maps_text = self.ui.lineEdit_5_4.text.strip()
        output_path = self.ui.lineEdit_5_6.text.strip()

        # 验证输入
        if not all([base_dir, pet_name, label_path, label_maps_text, output_path]):
            slicer.util.errorDisplay("All fields are required!")
            return
        
        if not os.path.isdir(base_dir):
            slicer.util.errorDisplay(f"Base directory does not exist: {base_dir}")
            return
        
        try:
            label_maps_list = [int(d.strip()) for d in label_maps_text.split(',')]
            label_maps = tuple(label_maps_list)
        except:
            slicer.util.errorDisplay("Please input the form of label map, such as: 128,128,128")
            return
        
        import pandas as pd

        # 准备结果数据结构
        results = {
            'Folder': [],
        }

        # 为每个标签添加列
        for label in label_maps:
            results[f'Label_{label}'] = []
        
        # 开始处理        
        try:
            # 获取所有子文件夹名
            subdirs = [d for d in os.listdir(base_dir) 
                    if os.path.isdir(os.path.join(base_dir, d))]
            subdirs.sort()
            
            if not subdirs:
                slicer.util.errorDisplay("no subdirs", windowTitle="Error dir path")
                return
            
            # 初始化进度条
            total_steps = len(subdirs)
            self.ui.progressBar5.setValue(0)
            self.ui.progressBar5.setMaximum(100)
            
            # 遍历每个子文件夹
            for i, subdir in enumerate(subdirs):
                # 计算进度百分比
                progress_value = int((i + 1) / total_steps * 100)
                self.updateProgress5(progress_value, f"Processing {i+1}/{total_steps}: {subdir}")
                
                subdir_path = os.path.join(base_dir, subdir)
                
                # 查找指定文件
                pet_path = os.path.join(subdir_path, pet_name)
                
                if not os.path.isfile(pet_path):
                    print(f"PET file not found: {pet_path}")
                    continue
                if not os.path.isfile(label_path):
                    print(f"Label file not found: {label_path}")
                    continue

                # 创建临时文件保存配准结果
                with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                try:
                    # 配准标签图像到PET图像空间
                    registered_data, affine = register_and_save(
                        original_img_path=label_path,
                        refer_img_path=pet_path,
                        output_path=temp_path,
                        method='vectorized',
                        interpolation_order=0,
                        dtype=np.float32
                    )
                    
                    # 计算SUVR
                    suvr_result = suvr_compute(temp_path, pet_path, label_maps)
                    
                    # 添加结果
                    results['Folder'].append(subdir)
                    
                    # 添加每个标签的值
                    for label in label_maps:
                        # 使用正确的键名，与suvr_compute返回的键一致
                        label_key = f'Label{label}'
                        if label_key in suvr_result:
                            results[f'Label_{label}'].append(suvr_result[label_key])
                        else:
                            results[f'Label_{label}'].append(None)  # 或者0，根据需求
                    
                except Exception as e:
                    print(f"Error processing {subdir}: {e}")
                    # 添加空值以保持数据对齐
                    results['Folder'].append(subdir)
                    for label in label_maps:
                        results[f'Label_{label}'].append(None)
                finally:
                    # 清理临时文件
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

            # 保存结果到Excel文件
            df = pd.DataFrame(results)
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 使用openpyxl引擎以支持格式设置
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='SUVr_Results')
                
                # 获取工作簿和工作表对象进行格式设置
                workbook = writer.book
                worksheet = writer.sheets['SUVr_Results']
                
                # 设置列宽
                worksheet.column_dimensions['A'].width = 20  # Folder列
                
                # 设置标签列的宽度
                for col_idx, col_name in enumerate(df.columns[1:], 2):  # 从第2列开始
                    col_letter = chr(64 + col_idx)  # A=65, B=66, etc.
                    worksheet.column_dimensions[col_letter].width = 12

            self.updateProgress5(100, "SUVr computation complete!")
            slicer.util.infoDisplay(f"SUVr calculation result is saved to:\n{output_path}")
            
        except Exception as e:
            slicer.util.errorDisplay(f"Error during processing: {str(e)}")
            import traceback
            traceback.print_exc()

    def updateProgress5(self, value, message):
        """更新进度辅助函数"""
        self.ui.progressBar5.setValue(value)
        self.ui.progressBar5.setFormat(message)
        slicer.app.processEvents()



    def onNodeSelected(self, node, load_image_func):
        if not node:
            return
        print("Selected node:", node.GetName())

        # 获取存储节点（StorageNode）
        storage_node = node.GetStorageNode()
        filepath = storage_node.GetFullNameFromFileName()
        
        # 设置所有切片视图的背景Volume
        layoutManager = slicer.app.layoutManager()
        for sliceViewName in ["Red", "Yellow", "Green"]:
            sliceLogic = layoutManager.sliceWidget(sliceViewName).sliceLogic()
            compositeNode = sliceLogic.GetSliceCompositeNode()
            compositeNode.SetBackgroundVolumeID(node.GetID())
            slicer.util.resetSliceViews()

        if filepath:
            load_image_func(filepath)  # 调用传入的加载函数

    def onNodeSelected_yield(self, node, load_image_func):
        if not node:
            return
        print("Selected node:", node.GetName())

        # 获取存储节点（StorageNode）
        storage_node = node.GetStorageNode()
        filepath = storage_node.GetFullNameFromFileName()
        print("Selected file:", filepath)

        if filepath:
            load_image_func(filepath)  # 调用传入的加载函数

    def onDicom2Nifit_CT(self):        
        ct_dicom_dir = self.ui.lineEdit1_1_Dicom2Nifit.text.strip()
        output_path = self.ui.lineEdit1_2_Dicom2Nifit.text.strip()
        output_name = self.ui.lineEdit1_3_Dicom2Nifit.text.strip()

        if not ct_dicom_dir:
            slicer.util.errorDisplay("Please provide a right path.")
            return

        with slicer.util.tryWithErrorDisplay("nDicom2Nifit_CT failed.", waitCursor=True):
            self.logic.runDicom2Nifit_CT(ct_dicom_dir, output_path, output_name)

    def onDicom2Nifit_PET(self):        
        pet_dicom_dir = self.ui.lineEdit1_1_Dicom2Nifit_pet.text.strip()
        output_path = self.ui.lineEdit1_2_Dicom2Nifit_pet.text.strip()
        output_name = self.ui.lineEdit1_3_Dicom2Nifit_pet.text.strip()

        if not pet_dicom_dir:
            slicer.util.errorDisplay("Please provide a right path.")
            return

        with slicer.util.tryWithErrorDisplay("nDicom2Nifit_PET failed.", waitCursor=True):
            self.logic.runDicom2Nifit_PET(pet_dicom_dir, output_path, output_name)

    def onSkullStrip(self):
        if not self.logic.image_skull:
            slicer.util.errorDisplay("Please load image.")
            return
        
        output_name = self.ui.lineEdit1_1_skull.text.strip()
        output_label_name = self.ui.lineEdit1_2_skull.text.strip()

        if not output_name:
            slicer.util.errorDisplay("Please provide a name for the output volume.")
            return

        with slicer.util.tryWithErrorDisplay("RigidRegistration failed.", waitCursor=True):
            self.logic.runSkullStrip(output_name, output_label_name)

    def onSkullStrip_mask(self):
        if not self.logic.image_skull_pet and not self.logic.image_skull_mask:
            slicer.util.errorDisplay("Please load image.")
            return
        
        output_name = self.ui.lineEdit1_1_skull_pet.text.strip()

        if not output_name:
            slicer.util.errorDisplay("Please provide a name for the output volume.")
            return

        with slicer.util.tryWithErrorDisplay("RigidRegistration failed.", waitCursor=True):
            self.logic.runSkullStrip_pet(output_name)

    def onCTclip(self):
        minimum_str = self.ui.lineEdit_min_CTclip.text.strip()
        maximum_str = self.ui.lineEdit_max_CTclip.text.strip()
        output_name = self.ui.lineEdit1_1_CTclip.text.strip()
        normalize = self.ui.CTClipNormalizeComboBox.currentText

        # 转换为数字
        try:
            minimum = float(minimum_str)
        except ValueError:
            slicer.util.errorDisplay(f"Invalid minimum value: {minimum_str}")
            return
            
        try:
            maximum = float(maximum_str)
        except ValueError:
            slicer.util.errorDisplay(f"Invalid maximum value: {maximum_str}")
            return

        if not output_name:
            slicer.util.errorDisplay("Please provide a name for the output volume.")
            return

        with slicer.util.tryWithErrorDisplay("CT Clip failed.", waitCursor=True):
            self.logic.runCTclip(minimum, maximum, output_name, normalize)

    def onRigidRegister(self):
        if not self.logic.image1_1 or not self.logic.image1_2:
            slicer.util.errorDisplay("Please load both fixed and moving images.")
            return
        
        output_name = self.ui.lineEdit1_1.text.strip()
        output_field_name = self.ui.lineEdit1_2.text.strip()
        interpolation_mode = self.ui.interpolationComboBox_rigid.currentText

        if not output_name:
            slicer.util.errorDisplay("Please provide a name for the output volume.")
            return

        with slicer.util.tryWithErrorDisplay("RigidRegistration failed.", waitCursor=True):
            self.logic.runRigidRegistration(output_name, output_field_name, interpolation_mode)

    def onRigidRegister_field(self):
        if not self.logic.image1_3:
            slicer.util.errorDisplay("Please load PET image.")
            return
        if not self.logic.image1_4:
            slicer.util.errorDisplay("Please load yield.")
            return
        
        output_name = self.ui.lineEdit1_1_rigid_pet.text.strip()

        if not output_name:
            slicer.util.errorDisplay("Please provide a name for the output volume.")
            return

        with slicer.util.tryWithErrorDisplay("RigidRegistration failed.", waitCursor=True):
            self.logic.runRigidRegistration_field(output_name)

    def onSpaceRegister(self):
        if not self.logic.image2_1 or not self.logic.image2_2:
            slicer.util.errorDisplay("Please load both fixed and moving images.")
            return
        
        interpolationComboBox_mode = self.ui.interpolationComboBox_spaceRegister.currentText
        
        resolution_text = self.ui.lineEdit2_1.text.strip()
        
        dimensions_text = self.ui.lineEdit2_2.text.strip()

        try:
            dimensions = [int(d) for d in dimensions_text.split(',')]
            if len(dimensions) != 3:
                raise ValueError
            in_shape = tuple(dimensions)
            print("dimensions:", in_shape)

            resolution = [int(d) for d in resolution_text.split(',')]
            if len(resolution) != 3:
                raise ValueError
            resolutions = tuple(resolution)
            print("resolution:", resolutions)
        except:
            slicer.util.errorDisplay("Please input the form of resolution, such as: 128,128,128")
            return
        
        output_name = self.ui.lineEdit2_3.text.strip()
        if not output_name:
            slicer.util.errorDisplay("Please provide a name for the output volume.")
            return

        with slicer.util.tryWithErrorDisplay("SpaceRegistration failed.", waitCursor=True):
            self.logic.runSpaceRegistration(in_shape=in_shape, resolution=resolutions, output_name=output_name, interpolationComboBox_mode=interpolationComboBox_mode)

    def onSynRegister(self):
        if not self.logic.image3_1 or not self.logic.image3_2:
            slicer.util.errorDisplay("Please load both fixed and moving images.")
            return
        
        output_name = self.ui.lineEdit3_1.text.strip()
        output_field_name = self.ui.lineEdit3_1_field.text.strip()

        if not output_name:
            slicer.util.errorDisplay("Please provide a name for the output volume.")
            return

        with slicer.util.tryWithErrorDisplay("SynRegistration failed.", waitCursor=True):
            self.logic.runSynRegistration(output_name, output_field_name)

    def onSynRegister_field(self):
        if not self.logic.image3_3:
            slicer.util.errorDisplay("Please load synthmorph PET images.")
            return
        
        yield_path = self.ui.lineEdit1_13_syn_pet.text.strip()
        output_name = self.ui.lineEdit1_1_syn_pet.text.strip()
        interpolation_mode = self.ui.interpolationComboBox_synthmorph_apply.currentText

        if not output_name:
            slicer.util.errorDisplay("Please provide a name for the output volume.")
            return

        with slicer.util.tryWithErrorDisplay("SynRegistration failed.", waitCursor=True):
            self.logic.runSynRegistration_field(yield_path, output_name, interpolation_mode)

    def onDiceCompute(self):
        if not self.logic.image4_1 or not self.logic.image4_2:
            slicer.util.errorDisplay("Please load both label1 and label2 images.")
            return
        
        labels_text = self.ui.lineEdit4_1.text.strip()

        labels = [int(d) for d in labels_text.split(',')]

        with slicer.util.tryWithErrorDisplay("Dice compute failed.", waitCursor=True):
            self.logic.runDiceCompute(labels)

    def onSuvrCompute(self):
        if not self.logic.image5_1 or not self.logic.image5_2:
            slicer.util.errorDisplay("Please load both pet and label images.")
            return
        
        labels_text = self.ui.lineEdit5_1.text.strip()

        labels = [int(d) for d in labels_text.split(',')]

        with slicer.util.tryWithErrorDisplay("SUVr compute failed.", waitCursor=True):
            self.logic.runSuvrCompute(labels)

    def onSuvrMapping(self):
        if not self.logic.image5_1_mapping or not self.logic.image5_2_mapping:
            slicer.util.errorDisplay("Please load both pet and label images.")
            return
        
        output_path = self.ui.lineEdit_mapping.text.strip()


        with slicer.util.tryWithErrorDisplay("SUVr compute failed.", waitCursor=True):
            self.logic.runSuvrMapping(output_path)



#
# SynCTLogic
#


class SynCTLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

        self.image1_1 = None
        self.image1_2 = None
        self.image1_3 = None
        self.image1_4 = None
        self.image2_1 = None
        self.image2_2 = None
        self.image3_1 = None
        self.image3_2 = None
        self.image3_3 = None
        self.image4_1 = None
        self.image4_2 = None
        self.image5_1 = None
        self.image5_2 = None
        self.image5_1_mapping = None
        self.image5_2_mapping = None
        self.image_skull = None

        self.filepath1_1 = None
        self.filepath1_2 = None
        self.filepath1_3 = None
        self.filepath1_4 = None
        self.filepath2_1 = None
        self.filepath2_2 = None
        self.filepath3_1 = None
        self.filepath3_2 = None
        self.filepath3_3 = None
        self.filepath4_1 = None
        self.filepath4_2 = None
        self.filepath5_1 = None
        self.filepath5_2 = None
        self.filepath5_1_mapping = None
        self.filepath5_2_mapping = None
        self.filepath_skull = None

    def loadImage(self, filepath_skull: str) -> None:
        """Load the image from the specified file path."""
        self.filepath_skull = filepath_skull
        # Load the image using Slicer utility function
        self.image_skull = slicer.util.loadVolume(filepath_skull)
        if self.image_skull:
            print(f"Image loaded successfully: {self.image_skull.GetName()}")
        else:
            print("Failed to load image.")

    def loadImage_skull_pet(self, filepath_skull_pet: str) -> None:
        """Load the image from the specified file path."""
        self.filepath_skull_pet = filepath_skull_pet
        # Load the image using Slicer utility function
        self.image_skull_pet = slicer.util.loadVolume(filepath_skull_pet)
        if self.image_skull_pet:
            print(f"Image loaded successfully: {self.image_skull_pet.GetName()}")
        else:
            print("Failed to load image.")

    def loadImage_skull_mask(self, filepath_skull_mask: str) -> None:
        """Load the image from the specified file path."""
        self.filepath_skull_mask = filepath_skull_mask
        # Load the image using Slicer utility function
        self.image_skull_mask = slicer.util.loadVolume(filepath_skull_mask)
        if self.image_skull_mask:
            print(f"Image loaded successfully: {self.image_skull_mask.GetName()}")
        else:
            print("Failed to load image.")

    def loadImage_CTclip(self, filepath_ct: str) -> None:
        """Load the image from the specified file path."""
        self.filepath_ct = filepath_ct
        # Load the image using Slicer utility function
        self.image_ct = slicer.util.loadVolume(filepath_ct)
        if self.image_ct:
            print(f"Image loaded successfully: {self.image_ct.GetName()}")
        else:
            print("Failed to load image.")

    def loadImage1_1(self, filepath1_1: str) -> None:
        """Load the image from the specified file path."""
        self.filepath1_1 = filepath1_1
        # Load the image using Slicer utility function
        self.image1_1 = slicer.util.loadVolume(filepath1_1)
        if self.image1_1:
            print(f"Image loaded successfully: {self.image1_1.GetName()}")
        else:
            print("Failed to load image.")

    def loadImage1_2(self, filepath1_2: str) -> None:
        """Load the image from the specified file path."""
        self.filepath1_2 = filepath1_2
        # Load the image using Slicer utility function
        self.image1_2 = slicer.util.loadVolume(filepath1_2)
        if self.image1_2:
            logging.info(f"Image loaded successfully: {self.image1_2.GetName()}")
        else:
            logging.error("Failed to load image.")

    def loadImage1_3(self, filepath1_3: str) -> None:
        """Load the image from the specified file path."""
        self.filepath1_3 = filepath1_3
        # Load the image using Slicer utility function
        self.image1_3 = slicer.util.loadVolume(filepath1_3)
        if self.image1_3:
            logging.info(f"Image loaded successfully: {self.image1_3.GetName()}")
        else:
            logging.error("Failed to load image.")

    def loadDeformationField(self, filepath1_4: str):
        """加载形变场变换"""
        self.filepath1_4 = filepath1_4
        print(f"Loading yield from: {filepath1_4}")
        # Load the image using Slicer utility function
        self.image1_4 = slicer.util.loadTransform(filepath1_4)
        if self.image1_4:
            logging.info(f"Yield loaded successfully: {self.image1_4.GetName()}")
        else:
            logging.error("Failed to load yield.")

    def loadImage2_1(self, filepath2_1: str) -> None:
        """Load the image from the specified file path."""
        self.filepath2_1 = filepath2_1
        # Load the image using Slicer utility function
        self.image2_1 = slicer.util.loadVolume(filepath2_1)
        if self.image2_1:
            logging.info(f"Image loaded successfully: {self.image2_1.GetName()}")
        else:
            logging.error("Failed to load image.")

    def loadImage2_2(self, filepath2_2: str) -> None:
        """Load the image from the specified file path."""
        self.filepath2_2 = filepath2_2
        # Load the image using Slicer utility function
        self.image2_2 = slicer.util.loadVolume(filepath2_2)
        if self.image2_2:
            logging.info(f"Image loaded successfully: {self.image2_2.GetName()}")
        else:
            logging.error("Failed to load image.")

    def loadImage3_1(self, filepath3_1: str) -> None:
        """Load the image from the specified file path."""
        self.filepath3_1 = filepath3_1
        # Load the image using Slicer utility function
        self.image3_1 = slicer.util.loadVolume(filepath3_1)
        if self.image3_1:
            logging.info(f"Image loaded successfully: {self.image3_1.GetName()}")
        else:
            logging.error("Failed to load image.")

    def loadImage3_2(self, filepath3_2: str) -> None:
        """Load the image from the specified file path."""
        self.filepath3_2 = filepath3_2
        # Load the image using Slicer utility function
        self.image3_2 = slicer.util.loadVolume(filepath3_2)
        if self.image3_2:
            logging.info(f"Image loaded successfully: {self.image3_2.GetName()}")
        else:
            logging.error("Failed to load image.")

    def loadImage3_3(self, filepath3_3: str) -> None:
        """Load the image from the specified file path."""
        self.filepath3_3 = filepath3_3
        # Load the image using Slicer utility function
        self.image3_3 = slicer.util.loadVolume(filepath3_3)
        if self.image3_3:
            logging.info(f"Image loaded successfully: {self.image3_3.GetName()}")
        else:
            logging.error("Failed to load image.")

    def loadImage4_1(self, filepath4_1: str) -> None:
        """Load the image from the specified file path."""
        self.filepath4_1 = filepath4_1
        # Load the image using Slicer utility function
        self.image4_1 = slicer.util.loadVolume(filepath4_1)
        if self.image4_1:
            logging.info(f"Image loaded successfully: {self.image4_1.GetName()}")
        else:
            logging.error("Failed to load image.")

    def loadImage4_2(self, filepath4_2: str) -> None:   
        """Load the image from the specified file path."""
        self.filepath4_2 = filepath4_2
        # Load the image using Slicer utility function
        self.image4_2 = slicer.util.loadVolume(filepath4_2)
        if self.image4_2:
            logging.info(f"Image loaded successfully: {self.image4_2.GetName()}")
        else:
            logging.error("Failed to load image.")

    def loadImage5_1(self, filepath5_1: str) -> None:
        """Load the image from the specified file path."""
        self.filepath5_1 = filepath5_1
        # Load the image using Slicer utility function
        self.image5_1 = slicer.util.loadVolume(filepath5_1)
        if self.image5_1:
            logging.info(f"Image loaded successfully: {self.image5_1.GetName()}")
        else:
            logging.error("Failed to load image.")

    def loadImage5_2(self, filepath5_2: str) -> None:
        """Load the image from the specified file path."""
        self.filepath5_2 = filepath5_2
        # Load the image using Slicer utility function
        self.image5_2 = slicer.util.loadVolume(filepath5_2)
        if self.image5_2:
            logging.info(f"Image loaded successfully: {self.image5_2.GetName()}")
        else:
            logging.error("Failed to load image.")

    def loadImage5_1_mapping(self, filepath5_1_mapping: str) -> None:
        """Load the image from the specified file path."""
        self.filepath5_1_mapping = filepath5_1_mapping
        # Load the image using Slicer utility function
        self.image5_1_mapping = slicer.util.loadVolume(filepath5_1_mapping)
        if self.image5_1_mapping:
            logging.info(f"Image loaded successfully: {self.image5_1_mapping.GetName()}")
        else:
            logging.error("Failed to load image.")

    def loadImage5_2_mapping(self, filepath5_2_mapping: str) -> None:
        """Load the image from the specified file path."""
        self.filepath5_2_mapping = filepath5_2_mapping
        # Load the image using Slicer utility function
        self.image5_2_mapping = slicer.util.loadVolume(filepath5_2_mapping)
        if self.image5_2:
            logging.info(f"Label loaded successfully: {self.image5_2_mapping.GetName()}")
        else:
            logging.error("Failed to load image.")

    def renameNode(node, new_name):
        """安全重命名节点，避免名称冲突"""
        if not node:
            return False
        
        # 检查名称是否已存在
        existing_node = slicer.util.getNode(new_name)
        if existing_node and existing_node != node:
            print(f"名称 {new_name} 已被占用，无法重命名")
            return False
        
        node.SetName(new_name)
        return True

    def runSkullStrip(self, output_name: str, output_label_name) -> None:
        if not self.image_skull:
            raise ValueError("Image is not loaded.")

        # Create output volume node
        self.rigidRegisteredVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", output_name)
        self.labelVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", output_label_name)

        # 运行 SwissSkullStripper
        parameters = {
            "patientVolume": self.image_skull.GetID(),
            "patientOutputVolume": self.rigidRegisteredVolumeNode.GetID(),
            "patientMaskLabel": self.labelVolumeNode.GetID(),
        }
        slicer.cli.run(slicer.modules.swissskullstripper, None, parameters)

        # 查看结果
        slicer.util.setSliceViewerLayers(background=self.rigidRegisteredVolumeNode)


    def runDicom2Nifit_CT(self, ct_dicom_dir: str, output_path: str, output_name: str) -> None:
        """
        将CT的DICOM文件转换为NIfTI格式
        
        Args:
            ct_dicom_dir: 输入DICOM文件目录路径
            output_path: 输出目录路径
            output_name: 输出NIfTI文件名（不包含扩展名）
        """
        try:
            # 将字符串路径转换为Path对象
            nii_out_path = plb.Path(output_path)
            
            print(f'ct_dicom_dir: {ct_dicom_dir}')
            print(f'output_path: {output_path}')
            
            # 确保输出目录存在
            nii_out_path.mkdir(parents=True, exist_ok=True)
            
            # 使用临时目录进行转换
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = plb.Path(tmp_dir)
                
                print("开始DICOM到NIfTI转换...")
                
                # 将DICOM目录转换为NIfTI
                dicom2nifti.convert_directory(ct_dicom_dir, str(tmp_path), compression=False, reorient=True)
                
                # 查找生成的NIfTI文件
                nii_files = list(tmp_path.glob('*.nii'))
                if not nii_files:
                    # 也检查.nii.gz文件
                    nii_files = list(tmp_path.glob('*.nii.gz'))
                
                if not nii_files:
                    # 列出临时目录中的所有文件用于调试
                    all_files = list(tmp_path.glob('*'))
                    print(f"临时目录中的文件: {[f.name for f in all_files]}")
                    raise FileNotFoundError("未找到转换后的NIfTI文件")
                
                # 获取第一个NIfTI文件
                nii_file = nii_files[0]
                # print(f"找到转换文件: {nii_file.name}")
                
                # 构建输出文件路径
                output_file = nii_out_path / f'{output_name}.nii'
                
                # 复制文件到目标位置
                shutil.copy(nii_file, output_file)

                self.displayVolumeInSlicer(str(output_file), output_name)
                
                print(f"转换成功: {output_file}")
                
        except Exception as e:
            print(f"转换失败: {str(e)}")
            raise

    def displayVolumeInSlicer(self, file_path: str, volume_name: str) -> None:
        """
        在3D Slicer中显示体积文件
        """
        try:
            # 显示进度信息
            slicer.util.showStatusMessage(f"正在加载体积: {volume_name}...")
            
            # 加载体积文件
            volume_node = slicer.util.loadVolume(file_path)
            
            if volume_node:
                # 重命名节点
                volume_node.SetName(volume_name)
                
                # 设置显示属性（针对CT）
                display_node = volume_node.GetDisplayNode()
                if display_node:
                    display_node.SetWindow(400)  # 窗口宽度
                    display_node.SetLevel(40)    # 窗口中心
                
                # 切换到合适的布局
                layout_manager = slicer.app.layoutManager()
                layout_manager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
                
                # 显示成功消息
                slicer.util.showStatusMessage(f"成功加载: {volume_name}", 3000)
                # print(f"体积 '{volume_name}' 已显示在3D Slicer中")
                
            else:
                slicer.util.errorDisplay(f"无法加载体积文件: {file_path}")
                
        except Exception as e:
            error_msg = f"显示体积失败: {str(e)}"
            slicer.util.errorDisplay(error_msg)
            print(error_msg)

    def conv_time(self, time_str):
        return (float(time_str[:2]) * 3600 + float(time_str[2:4]) * 60 + float(time_str[4:13]))

    def calculate_suv_factor(self, dcm_path):
        ds = pydicom.read_file(str(dcm_path))
        total_dose = ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
        start_time = ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
        half_life = ds.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife
        acq_time = ds.AcquisitionTime
        weight = ds.PatientWeight
        time_diff = self.conv_time(acq_time) - self.conv_time(start_time)
        act_dose = total_dose * 0.5 ** (time_diff / half_life)
        suv_factor = 1000 * (weight) / act_dose
        return suv_factor
    
    def convert_pet(self, pet, suv_factor):
        affine = pet.affine
        pet_data = pet.get_fdata()
        pet_suv_data = (pet_data * suv_factor).astype(np.float32)
        pet_suv = nib.Nifti1Image(pet_suv_data, affine)
        return pet_suv
    
    def runDicom2Nifit_PET(self, pet_dicom_dir: str, output_path: str, output_name: str) -> None:
        try:
            # 将字符串路径转换为Path对象
            pet_dicom_dir = plb.Path(pet_dicom_dir)
            nii_out_path = plb.Path(output_path)
            
            print(f'ct_dicom_dir: {pet_dicom_dir}')
            print(f'output_path: {output_path}')
            
            # 确保输出目录存在
            nii_out_path.mkdir(parents=True, exist_ok=True)

            first_pt_dcm = next(pet_dicom_dir.glob('*'))
            suv_corr_factor = self.calculate_suv_factor(first_pt_dcm)
            
            # 使用临时目录进行转换
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = plb.Path(tmp_dir)
                
                print("开始DICOM到NIfTI转换...")
                
                # 将DICOM目录转换为NIfTI
                dicom2nifti.convert_directory(pet_dicom_dir, str(tmp_path), compression=False, reorient=True)
                
                nii = next(tmp_path.glob('*nii'))
                suv_pet_nii = self.convert_pet(nib.load(nii), suv_factor=suv_corr_factor)
                nib.save(suv_pet_nii, nii_out_path / f'{output_name}.nii')
                pet_path = nii_out_path / f'{output_name}.nii'
                
                print(f"转换成功: {pet_path}")

                self.displayVolumeInSlicer(pet_path, output_name)
                
        except Exception as e:
            print(f"转换失败: {str(e)}")
            raise




    # def runSkullStrip(self, output_name: str, output_label_name: str) -> None:
    #     if not self.image_skull:
    #         raise ValueError("Image is not loaded.")

    #     # 创建临时文件路径
    #     import tempfile
    #     import os
        
    #     # 创建临时目录
    #     temp_dir = tempfile.mkdtemp()
    #     input_path = os.path.join(temp_dir, "input.nii.gz")
    #     output_path = os.path.join(temp_dir, "output.nii.gz")
    #     mask_path = os.path.join(temp_dir, "mask.nii.gz")
        
    #     try:
    #         # 保存输入图像到临时文件
    #         slicer.util.saveNode(self.image_skull, input_path)
            
    #         # 构建 mri_synthstrip 命令
    #         mri_synthstrip_script = os.path.join(os.path.dirname(__file__), "./tools/mri_synthstrip.py")
    #         mri_synthstrip_model = os.path.join(os.path.dirname(__file__), "./synthstrip.1.pt")
    #         print(f"mri_synthstrip script path: {mri_synthstrip_script}")
            
    #         if not os.path.exists(mri_synthstrip_script):
    #             raise FileNotFoundError(f"mri_synthstrip.py not found at: {mri_synthstrip_script}")
            
    #         # 设置命令参数
    #         import subprocess
    #         cmd = [
    #             r".\bin\PythonSlicer.exe",
    #             "./tools/mri_synthstrip.py",
    #             "-i", input_path,
    #             "-o", output_path,
    #             "-m", mask_path,
    #             "--model", mri_synthstrip_model,
    #         ]
            
    #         # 执行命令
    #         print(f"Running command: {' '.join(cmd)}")
    #         result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
            
    #         if result.returncode != 0:
    #             raise RuntimeError(f"mri_synthstrip failed: {result.stderr}")
            
    #         print(f"mri_synthstrip output: {result.stdout}")
            
    #         # 加载结果
    #         if os.path.exists(output_path):
    #             self.rigidRegisteredVolumeNode = slicer.util.loadVolume(output_path, properties={'name': output_name})
    #         else:
    #             raise FileNotFoundError(f"Output file not found: {output_path}")
            
    #         if os.path.exists(mask_path):
    #             self.labelVolumeNode = slicer.util.loadVolume(mask_path, properties={'name': output_label_name})
    #         else:
    #             raise FileNotFoundError(f"Mask file not found: {mask_path}")
            
    #         # 查看结果
    #         slicer.util.setSliceViewerLayers(background=self.rigidRegisteredVolumeNode)
            
    #     except Exception as e:
    #         slicer.util.errorDisplay(f"Skull stripping failed: {str(e)}")
    #         raise
    #     finally:
    #         # 清理临时文件
    #         import shutil
    #         try:
    #             shutil.rmtree(temp_dir)
    #         except:
    #             pass

    def runSkullStrip_pet(self, output_name: str) -> None:
        """应用mask到PET图像，只保留mask不为0的区域"""
        try:
            # 基本检查
            if not self.image_skull_pet or not self.image_skull_mask:
                raise ValueError("Please load both PET image and mask first")
            
            # 获取numpy数组
            pet_data = slicer.util.arrayFromVolume(self.image_skull_pet)
            mask_data = slicer.util.arrayFromVolume(self.image_skull_mask)
            
            # 简单形状检查
            if pet_data.shape != mask_data.shape:
                raise ValueError(f"Shape mismatch: PET {pet_data.shape} vs Mask {mask_data.shape}")
            
            # 应用mask
            result_data = pet_data * (mask_data > 0)
            
            # 创建输出体积 - 使用正确的参数名称
            self.rigidRegisteredVolumeNode = slicer.util.addVolumeFromArray(
                result_data, 
                name=output_name
            )
            
            # 手动设置几何信息
            self.rigidRegisteredVolumeNode.SetSpacing(self.image_skull_pet.GetSpacing())
            self.rigidRegisteredVolumeNode.SetOrigin(self.image_skull_pet.GetOrigin())
            
            # 复制变换矩阵
            ijkToRAS = vtk.vtkMatrix4x4()
            self.image_skull_pet.GetIJKToRASMatrix(ijkToRAS)
            self.rigidRegisteredVolumeNode.SetIJKToRASMatrix(ijkToRAS)
            
            print("PET skull stripping completed successfully")
            slicer.util.setSliceViewerLayers(background=self.rigidRegisteredVolumeNode)
            
        except Exception as e:
            slicer.util.errorDisplay(f"Error: {str(e)}")

    # 加载NIfTI文件
    def load_nifti(self, file_path):
        nifti_img = nib.load(file_path)
        data = nifti_img.get_fdata()
        affine = nifti_img.affine
        header = nifti_img.header
        return data, affine, header


    # 设置阈值并进行Min-Max归一化
    def threshold_and_normalize(self, data, min_val: float, max_val: float, normalize: str):
        # 设置阈值
        data = np.clip(data, min_val, max_val)

        if normalize == 'True':
            # Min-Max归一化
            data_min = np.min(data)
            data_max = np.max(data)
            if data_max - data_min == 0:
                raise ValueError("The data has no variation; min and max values are equal.")
            normalized_data = (data - data_min) / (data_max - data_min)
        else:
            normalized_data = data
        return normalized_data
    
    # 保存NIfTI文件
    def save_nifti(self, data, affine, header, output_file):
        nifti_img = nib.Nifti1Image(data, affine, header=header)
        nib.save(nifti_img, output_file)

    def runCTclip(self, minimum: float, maximum: float, output_name: str, normalize: str) -> None:
        # Create output volume node
        # print(f'filepath_ct: {self.filepath_ct}, minimum: {minimum}, maximum: {maximum}, output_name: {output_name}')

        data, affine, header = self.load_nifti(self.filepath_ct)
        normalized_data = self.threshold_and_normalize(data, minimum, maximum, normalize)

        output_dir = os.path.join(
            os.path.dirname(__file__),  # 当前脚本目录
            "tmp_data"
        )
        os.makedirs(output_dir, exist_ok=True)  # 确保目录存在
        output_file_path = os.path.join(output_dir, "{output_filename}.nii.gz".format(output_filename=output_name))

        self.save_nifti(normalized_data, affine, header, output_file_path)
        print(f"CTClip image saved to: {output_file_path}")

        self.ctclipVolumeNode = slicer.util.loadVolume(output_file_path, properties={'name': output_name})
        slicer.util.setSliceViewerLayers(background=self.ctclipVolumeNode)
        print("CT clip completed successfully.")

    def runRigidRegistration(self, output_name: str, output_field_name: str, interpolation_mode: str) -> None:
        print(f'interpolation_mode: {interpolation_mode}')
        if not self.image1_1 and not self.image1_2:
            raise ValueError("Fixed or moving image is not loaded.")

        # Create output volume node
        self.rigidRegisteredVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", output_name)
        self.rigidRegisteredFieldVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", output_field_name)

        # Set up BrainsFit parameters
        parameters = {
            "fixedVolume": self.image1_2.GetID(),
            "movingVolume": self.image1_1.GetID(),
            "outputVolume": self.rigidRegisteredVolumeNode.GetID(),
            "outputTransform": self.rigidRegisteredFieldVolumeNode.GetID(),  # 添加形变场输出参数
            "useRigid": True,  # Enable rigid registration
            "useAffine": False,  # Disable affine registration
            "samplingPercentage": 0.01,  # Adjust for faster or more accurate results
            "initializeTransformMode": "useGeometryAlign",  # Initialize transform based on geometry
            # 添加明确的插值参数
            "interpolationMode": interpolation_mode,
        }

        # Run BrainsFit
        cliNode = slicer.cli.runSync(slicer.modules.brainsfit, None, parameters)

        if cliNode.GetStatusString() == "Completed":
            print("rigidRegistration completed successfully.")
            slicer.util.setSliceViewerLayers(background=self.rigidRegisteredVolumeNode)

        else:
            slicer.util.errorDisplay("rigidRegistration failed. Check the log for details.")

    def runRigidRegistration_field(self, output_name: str) -> None:
        if not self.image1_3 or not self.image1_4:
            raise ValueError("Fixed image or deformation transform is not loaded.")
        
        parameters = {
            "inputVolume": self.image1_3.GetID(),
            "referenceVolume": self.image1_3.GetID(),  # 可以用自己，或另一张图作为参考空间
            "outputVolume": slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", output_name).GetID(),
            "warpTransform": self.image1_4.GetID()
        }

        cliNode = slicer.cli.runSync(slicer.modules.brainsresample, None, parameters)
        outputNode = slicer.mrmlScene.GetNodeByID(parameters["outputVolume"])

        if cliNode.GetStatusString() == "Completed":
            print("Rigid registration applied successfully.")
            slicer.util.setSliceViewerLayers(background=outputNode)
        else:
            slicer.util.errorDisplay("Failed to apply transform. Check the log for details.")

    def runSpaceRegistration(self, in_shape, resolution, output_name, interpolationComboBox_mode) -> None:
        if not self.image2_1 or not self.image2_2:
            raise ValueError("Fixed or moving image is not loaded.")

        # Create output volume node
        # self.spaceRegisteredVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "spaceRegisteredVolume")

        # in_shape = (192,) * 3
        # out_dir = 'E:\\software\\3D Slicer\\Slicer 5.8.1\\slicer.org\\Extensions-33241\\TIPS\\TIPS\\data'
        out_dir = '.\\temp_data'
        # Threading.
        tf.config.threading.set_inter_op_parallelism_threads(12)
        tf.config.threading.set_intra_op_parallelism_threads(12)

        # Input data.
        print("self.filepath2_1:", self.filepath2_1)
        mov = sf.load_volume(self.filepath2_1)
        fix = sf.load_volume(self.filepath2_2)
        if not len(mov.shape) == len(fix.shape) == 3:
            sf.system.fatal('input images are not single-frame volumes')

        center = fix
        net_to_mov, mov_to_net = self.network_space(im=mov, shape=in_shape, voxsize=resolution, center=center)
        net_to_fix, fix_to_net = self.network_space(fix, shape=in_shape, voxsize=resolution)

        mov_to_ras = mov.geom.vox2world.matrix
        fix_to_ras = fix.geom.vox2world.matrix

        inputs = (
                self.transform(mov, net_to_mov, shape=in_shape, normalize=False, batch=True, interpolationComboBox_mode=interpolationComboBox_mode),
                self.transform(fix, net_to_fix, shape=in_shape, normalize=False, batch=True, interpolationComboBox_mode=interpolationComboBox_mode),
            )

        os.makedirs(out_dir, exist_ok=True)
        # print(f"输出目录的绝对路径: {os.path.abspath(out_dir)}")
        
        inp_1 = os.path.join(out_dir, f'{output_name}.nii')
        inp_2 = os.path.join(out_dir, f'fixed.nii')
        
        geom_1 = sf.ImageGeometry(in_shape, vox2world=mov_to_ras @ net_to_mov)
        geom_2 = sf.ImageGeometry(in_shape, vox2world=fix_to_ras @ net_to_fix)
        sf.Volume(inputs[0][0], geom_1).save(inp_1)
        sf.Volume(inputs[1][0], geom_2).save(inp_2)

        # Display the result
        # slicer.util.setSliceViewerLayers(background=self.spaceRegisteredVolumeNode)
        # Load the saved volume into 3D Slicer
        loadedVolumeNode = slicer.util.loadVolume(inp_1)
        
        # Optionally set the name of the loaded volume
        loadedVolumeNode.SetName(output_name)
        
        # Automatically select the volume in the slice viewers
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectionNode.SetReferenceActiveVolumeID(loadedVolumeNode.GetID())
        slicer.app.applicationLogic().PropagateVolumeSelection(0)


    def runSynRegistration(self, output_filename: str, output_field_name: str) -> None:
        if not self.image3_1 or not self.image3_2:
            raise ValueError("Fixed or moving image is not loaded.")
        
        import subprocess

        output_dir = os.path.join(
            os.path.dirname(__file__),  # 当前脚本目录
            "tmp_data"
        )
        os.makedirs(output_dir, exist_ok=True)  # 确保目录存在
        output_path = os.path.join(output_dir, "{output_filename}.nii.gz".format(output_filename=output_filename))
        output_field_path = os.path.join(output_dir, "{output_field_name}.nii.gz".format(output_field_name=output_field_name))

        cmd = [
            r".\bin\PythonSlicer.exe",  # 确保使用 Slicer 的 Python
            r".\slicer.org\Extensions-33241\PETCTREGISTER\PETCTREGISTER\mri_synthmorph\mri_synthmorph.py", 
            "register",
            self.filepath3_1,
            self.filepath3_2,
            "-m", "joint",
            "-o", output_path,
            "-t", output_field_path
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='latin-1')
            print(result.stdout)
            print(f'output_image_path: {output_path}')
            print(f'output_field_path: {output_field_path}')
        except subprocess.CalledProcessError as e:
            print("Command failed with error:")
            print(e.stderr)


    def runSynRegistration_field(self, yield_path: str, output_filename: str, interpolation_mode: str) -> None:
        if not self.image3_3:
            raise ValueError("Synthmorph PET image is not loaded.")
        if not os.path.exists(yield_path):
            raise ValueError(f"Yield file {yield_path} does not exist.")

        import subprocess

        output_dir = os.path.join(
            os.path.dirname(__file__),  # 当前脚本目录
            "tmp_data"
        )
        os.makedirs(output_dir, exist_ok=True)  # 确保目录存在
        output_path = os.path.join(output_dir, "{output_filename}.nii.gz".format(output_filename=output_filename))

        cmd = [
            r".\bin\PythonSlicer.exe",  # 确保使用 Slicer 的 Python
            r".\slicer.org\Extensions-33241\PETCTREGISTER\PETCTREGISTER\mri_synthmorph\mri_synthmorph.py", 
            "apply",
            yield_path,
            self.filepath3_3,
            output_path,
            "-m", interpolation_mode
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='latin-1')
            print(result.stdout)
            print(f'output_image_path: {output_path}')
        except subprocess.CalledProcessError as e:
            print("Command failed with error:")
            print(e.stderr)

        # import time
        # time.sleep(120)

        # # 加载 NIfTI 文件
        # loaded_volume = slicer.util.loadVolume(output_path)

        # # 获取切片视图的布局
        # layout_manager = slicer.app.layoutManager()

        # # 显示在三视图（红、黄、绿切片）中
        # red_slice = layout_manager.sliceWidget("Red")
        # yellow_slice = layout_manager.sliceWidget("Yellow")
        # green_slice = layout_manager.sliceWidget("Green")

        # # 设置每个切片的背景为加载的体数据
        # red_slice.sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(loaded_volume.GetID())
        # yellow_slice.sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(loaded_volume.GetID())
        # green_slice.sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(loaded_volume.GetID())

        # # 重置视图范围
        # slicer.util.resetSliceViews()

    def runDiceCompute(self, labels):
        if not self.image4_1 or not self.image4_2:
            raise ValueError("label1 or label2 image is not loaded.")

        import dice_calculate as dc

        dice_result = dc.dice_compute(self.filepath4_1, self.filepath4_2, labels=labels)
        print('Dice: %.4f +/- %.4f' % (dice_result[0], dice_result[1]))

    def runSuvrCompute(self, labels):
        if not self.image5_1 or not self.image5_2:
            raise ValueError("pet or label image is not loaded.")

        # 创建临时文件保存配准结果
        with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # 配准标签图像到PET图像空间
            registered_data, affine = register_and_save(
                original_img_path=self.filepath5_2,
                refer_img_path=self.filepath5_1,
                output_path=temp_path,
                method='vectorized',
                interpolation_order=0,
                dtype=np.float32
            )
            
            # 计算SUVR
            suvr_result = suvr_compute(temp_path, self.filepath5_1, labels)
            print("计算的SUVR结果为：", suvr_result)
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def runSuvrMapping(self, output_path):
        if not self.image5_1_mapping or not self.image5_2_mapping:
            raise ValueError("pet or label image is not loaded.")

        # 初始化归一化器
        normalizer = PETNormalizerWithRegistration()
        
        # 加载图像（请替换为实际路径）
        pet_path = self.filepath5_1_mapping
        ref_mask_path = self.filepath5_2_mapping  # 0为背景，1为小脑灰质
        
        if not normalizer.load_images(pet_path, ref_mask_path):
            return
        
        # 检查图像兼容性
        if not normalizer.check_image_compatibility():
            print("图像不兼容，进行配准...")
            # 进行配准（使用您提供的MATLAB代码方法）
            if not normalizer.register_mask_to_pet():
                print("配准失败，使用原始mask")
        else:
            print("图像兼容，跳过配准步骤")
        
        # 计算SUVR
        try:
            suvr_img, suvr_data = normalizer.calculate_suvr(use_registered_mask=True)
            
            # 保存结果
            normalizer.save_suvr_image(suvr_img, output_path)
            
        except Exception as e:
            print(f"SUVR计算错误: {e}")

    # space_register
    def network_space(self, im, shape, voxsize, center=None):
        """Construct transform from network space to the voxel space of an image.

        Constructs a coordinate transform from the space the network will operate
        in to the zero-based image index space. The network space has isotropic
        1-mm voxels, left-inferior-anterior (LIA) orientation, and no shear. It is
        centered on the field of view, or that of a reference image. This space is
        an indexed voxel space, not world space.

        Parameters
        ----------
        im : surfa.Volume
            Input image to construct the transform for.
        shape : (3,) array-like
            Spatial shape of the network space.
        center : surfa.Volume, optional
            Center the network space on the center of a reference image.

        Returns
        -------
        out : tuple of (3, 4) NumPy arrays
            Transform from network to input-image space and its inverse, thinking
            coordinates.

        """
        old = im.geom
        new = sf.ImageGeometry(
            shape=shape,
            voxsize=voxsize,
            rotation='LIA',
            center=old.center if center is None else center.geom.center,
            shear=None,
        )

        net_to_vox = old.world2vox @ new.vox2world
        vox_to_net = new.world2vox @ old.vox2world
        return net_to_vox.matrix, vox_to_net.matrix

    def is_affine_shape(self, shape):
        """
        Determine whether the given shape (single-batch) represents an N-dimensional affine matrix of
        shape (M, N + 1), with `N in (2, 3)` and `M in (N, N + 1)`.

        Parameters:
            shape: Tuple or list of integers excluding the batch dimension.
        """
        if len(shape) == 2 and shape[-1] != 1:
            self.validate_affine_shape(shape)
            return True
        return False


    def validate_affine_shape(self, shape):
        """
        Validate whether the input shape represents a valid affine matrix of shape (..., M, N + 1),
        where N is the number of dimensions, and M is N or N + 1. Throws an error if the shape is
        invalid.

        Parameters:
            shape: Tuple or list of integers.
        """
        ndim = shape[-1] - 1
        rows = shape[-2]
        if ndim not in (2, 3):
            raise ValueError(f'Affine matrix must be 2D or 3D, got {ndim}D')
        if rows not in (ndim, ndim + 1):
            raise ValueError(f'{ndim}D affine matrix must have {ndim} or {ndim + 1} rows, got {rows}.')

    def affine_to_dense_shift(self, matrix, shape, shift_center=True, warp_right=None):
        """
        Convert N-dimensional (ND) matrix transforms to dense displacement fields.

        Algorithm:
            1. Build and (optionally) shift grid to center of image.
            2. Apply matrices to each index coordinate.
            3. Subtract grid.

        Parameters:
            matrix: Affine matrix of shape (..., M, N + 1), where M is N or N + 1. Can have any batch
                dimensions.
            shape: ND shape of the output space.
            shift_center: Shift grid to image center.
            warp_right: Right-compose the matrix transform with a displacement field of shape
                (..., *shape, N), with batch dimensions broadcastable to those of `matrix`.

        Returns:
            Dense shift (warp) of shape (..., *shape, N).

        Notes:
            There used to be an argument for choosing between matrix ('ij') and Cartesian ('xy')
            indexing. Due to inconsistencies in how some functions and layers handled xy-indexing, we
            removed it in favor of default ij-indexing to minimize the potential for confusion.

        """
        if isinstance(shape, (tf.compat.v1.Dimension, tf.TensorShape)):
            shape = shape.as_list()

        if not tf.is_tensor(matrix) or not matrix.dtype.is_floating:
            matrix = tf.cast(matrix, tf.float32)

        # check input shapes
        ndims = len(shape)
        if matrix.shape[-1] != (ndims + 1):
            matdim = matrix.shape[-1] - 1
            raise ValueError(f'Affine ({matdim}D) does not match target shape ({ndims}D).')
        self.validate_affine_shape(matrix.shape)

        # coordinate grid
        mesh = (tf.range(s, dtype=matrix.dtype) for s in shape)
        if shift_center:
            mesh = (m - 0.5 * (s - 1) for m, s in zip(mesh, shape))
        mesh = [tf.reshape(m, shape=(-1,)) for m in tf.meshgrid(*mesh, indexing='ij')]
        mesh = tf.stack(mesh)  # N x nb_voxels
        out = mesh

        # optionally right-compose with warp field
        if warp_right is not None:
            if not tf.is_tensor(warp_right) or warp_right.dtype != matrix.dtype:
                warp_right = tf.cast(warp_right, matrix.dtype)
            flat_shape = tf.concat((tf.shape(warp_right)[:-1 - ndims], (-1, ndims)), axis=0)
            warp_right = tf.reshape(warp_right, flat_shape)  # ... x nb_voxels x N
            out += tf.linalg.matrix_transpose(warp_right)  # ... x N x nb_voxels

        # compute locations, subtract grid to obtain shift
        out = matrix[..., :ndims, :-1] @ out + matrix[..., :ndims, -1:]  # ... x N x nb_voxels
        out = tf.linalg.matrix_transpose(out - mesh)  # ... x nb_voxels x N

        # restore shape
        shape = tf.concat((tf.shape(matrix)[:-2], (*shape, ndims)), axis=0)
        return tf.reshape(out, shape)  # ... x in_shape x N


    def transformS(self, vol, loc_shift, interp_method='nearest', fill_value=None,
                shift_center=True, shape=None):
        """Apply affine or dense transforms to images in N dimensions.

        Essentially interpolates the input ND tensor at locations determined by
        loc_shift. The latter can be an affine transform or dense field of location
        shifts in the sense that at location x we now have the data from x + dx, so
        we moved the data.

        Parameters:
            vol: tensor or array-like structure  of size vol_shape or
                (*vol_shape, C), where C is the number of channels.
            loc_shift: Affine transformation matrix of shape (N, N+1) or a shift
                volume of shape (*new_vol_shape, D) or (*new_vol_shape, C, D),
                where C is the number of channels, and D is the dimensionality
                D = len(vol_shape). If the shape is (*new_vol_shape, D), the same
                transform applies to all channels of the input tensor.
            interp_method: 'linear' or 'nearest'.
            fill_value: Value to use for points sampled outside the domain. If
                None, the nearest neighbors will be used.
            shift_center: Shift grid to image center when converting affine
                transforms to dense transforms. Assumes the input and output spaces are identical.
            shape: ND output shape used when converting affine transforms to dense
                transforms. Includes only the N spatial dimensions. If None, the
                shape of the input image will be used. Incompatible with `shift_center=True`.

        Returns:
            Tensor whose voxel values are the values of the input tensor
            interpolated at the locations defined by the transform.

        Notes:
            There used to be an argument for choosing between matrix ('ij') and Cartesian ('xy')
            indexing. Due to inconsistencies in how some functions and layers handled xy-indexing, we
            removed it in favor of default ij-indexing to minimize the potential for confusion.

        Keywords:
            interpolation, sampler, resampler, linear, bilinear
        """
        if shape is not None and shift_center:
            raise ValueError('`shape` option incompatible with `shift_center=True`')

        # convert data type if needed
        ftype = tf.float32
        if not tf.is_tensor(vol) or not vol.dtype.is_floating:
            vol = tf.cast(vol, ftype)
        if not tf.is_tensor(loc_shift) or not loc_shift.dtype.is_floating:
            loc_shift = tf.cast(loc_shift, ftype)

        # convert affine to location shift (will validate affine shape)
        if self.is_affine_shape(loc_shift.shape):
            loc_shift = self.affine_to_dense_shift(loc_shift,
                                            shape=vol.shape[:-1] if shape is None else shape,
                                            shift_center=shift_center)

        # parse spatial location shape, including channels if available
        loc_volshape = loc_shift.shape[:-1]
        if isinstance(loc_volshape, (tf.compat.v1.Dimension, tf.TensorShape)):
            loc_volshape = loc_volshape.as_list()

        # volume dimensions
        nb_dims = len(vol.shape) - 1
        is_channelwise = len(loc_volshape) == (nb_dims + 1)
        assert loc_shift.shape[-1] == nb_dims, \
            'Dimension check failed for ne.utils.transform(): {}D volume (shape {}) called ' \
            'with {}D transform'.format(nb_dims, vol.shape[:-1], loc_shift.shape[-1])

        # location should be mesh and delta
        mesh = ne.utils.volshape_to_meshgrid(loc_volshape, indexing='ij')  # volume mesh
        for d, m in enumerate(mesh):
            if m.dtype != loc_shift.dtype:
                mesh[d] = tf.cast(m, loc_shift.dtype)
        loc = [mesh[d] + loc_shift[..., d] for d in range(nb_dims)]

        # if channelwise location, then append the channel as part of the location lookup
        if is_channelwise:
            loc.append(mesh[-1])

        # test single
        return ne.utils.interpn(vol, loc, interp_method=interp_method, fill_value=fill_value)


    def transform(self, im, trans, shape=None, normalize=False, batch=False, interpolationComboBox_mode='nearest'):
        """Apply a spatial transform to 3D image voxel data in dimensions.

        Applies a transformation matrix operating in zero-based index space or a
        displacement field to an image buffer.

        Parameters
        ----------
        im : surfa.Volume or NumPy array or TensorFlow tensor
            Input image to transform, without batch dimension.
        trans : array-like
            Transform to apply to the image. A matrix of shape (3, 4), a matrix
            of shape (4, 4), or a displacement field of shape (*space, 3),
            without batch dimension.
        shape : (3,) array-like, optional
            Output shape used for converting matrices to dense transforms. None
            means the shape of the input image will be used.
        normalize : bool, optional
            Min-max normalize the image intensities into the interval [0, 1].
        batch : bool, optional
            Prepend a singleton batch dimension to the output tensor.

        Returns
        -------
        out : float TensorFlow tensor
            Transformed image with a trailing feature dimension.

        """
        # Add singleton feature dimension if needed.
        if tf.rank(im) == 3:
            im = im[..., tf.newaxis]

        out = self.transformS(
            im, trans, fill_value=0, shift_center=False, shape=shape, interp_method=interpolationComboBox_mode
        )

        if normalize:
            out -= tf.reduce_min(out)
            out /= tf.reduce_max(out)

        if batch:
            out = out[tf.newaxis, ...]

        return out
    

# suvr mapping
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
import os

def deform_img_based_on_other_img(original_img_path, refer_img_path):
    """
    基于参考图像对原始图像进行空间变换
    
    参数:
        original_img_path: 原始图像路径
        refer_img_path: 参考图像路径
    
    返回:
        new_img_data: 变换后的图像数据
        refer_affine: 参考图像的仿射矩阵
    """
    # 加载图像
    refer_img = nib.load(refer_img_path)
    original_img = nib.load(original_img_path)
    
    # 获取图像数据
    refer_data = refer_img.get_fdata()
    original_data = original_img.get_fdata()
    
    # 获取仿射矩阵
    refer_affine = refer_img.affine
    original_affine = original_img.affine
    
    # 获取图像尺寸
    refer_x, refer_y, refer_z = refer_data.shape
    origin_x, origin_y, origin_z = original_data.shape
    
    # 初始化新图像
    new_img = np.zeros((refer_x, refer_y, refer_z))
    
    # 计算从参考图像空间到原始图像空间的变换矩阵
    transform_matrix = np.linalg.inv(original_affine) @ refer_affine
    
    # 创建坐标网格（更高效的方法）
    i, j, k = np.meshgrid(
        np.arange(refer_x), 
        np.arange(refer_y), 
        np.arange(refer_z),
        indexing='ij'
    )
    
    # 将坐标展平以便批量处理
    coords = np.stack([i.flatten(), j.flatten(), k.flatten(), np.ones(i.size)]).T
    
    # 应用变换矩阵
    physical_coords = coords @ refer_affine.T
    origin_coords = physical_coords @ np.linalg.inv(original_affine).T
    
    # 提取坐标并重塑
    origin_i = origin_coords[:, 0].reshape(refer_x, refer_y, refer_z)
    origin_j = origin_coords[:, 1].reshape(refer_x, refer_y, refer_z)
    origin_k = origin_coords[:, 2].reshape(refer_x, refer_y, refer_z)
    
    # 使用插值获取原始图像值（最近邻插值，与MATLAB代码一致）
    new_img = map_coordinates(
        original_data, 
        [origin_i, origin_j, origin_k], 
        order=0,  # 0=最近邻插值
        mode='constant', 
        cval=0.0
    )
    
    return new_img, refer_affine

class PETNormalizerWithRegistration:
    def __init__(self):
        self.pet_img = None
        self.ref_mask_img = None
        self.pet_data = None
        self.ref_mask_data = None
        self.registered_mask = None
        
    def load_images(self, pet_path, ref_mask_path):
        """加载PET图像和参考脑区mask"""
        try:
            # 加载PET图像
            self.pet_img = nib.load(pet_path)
            self.pet_data = self.pet_img.get_fdata()
            
            # 加载参考脑区mask
            self.ref_mask_img = nib.load(ref_mask_path)
            self.ref_mask_data = self.ref_mask_img.get_fdata()
            
            print(f"PET图像尺寸: {self.pet_data.shape}")
            print(f"PET图像分辨率: {self.pet_img.header.get_zooms()}")
            print(f"参考脑区尺寸: {self.ref_mask_data.shape}")
            print(f"参考脑区分辨率: {self.ref_mask_img.header.get_zooms()}")
            
            return True
            
        except Exception as e:
            print(f"图像加载错误: {e}")
            return False
    
    def check_image_compatibility(self):
        """检查图像兼容性"""
        pet_shape = self.pet_data.shape
        ref_shape = self.ref_mask_data.shape
        pet_affine = self.pet_img.affine
        ref_affine = self.ref_mask_img.affine
        
        # 检查尺寸是否一致
        shape_match = (pet_shape == ref_shape)
        
        # 检查仿射矩阵是否一致（允许小的数值差异）
        affine_match = np.allclose(pet_affine, ref_affine, atol=1e-3)
        
        # 检查分辨率
        pet_zooms = self.pet_img.header.get_zooms()
        ref_zooms = self.ref_mask_img.header.get_zooms()
        resolution_match = np.allclose(pet_zooms, ref_zooms, atol=0.1)
        
        print(f"图像尺寸匹配: {shape_match}")
        print(f"仿射矩阵匹配: {affine_match}")
        print(f"分辨率匹配: {resolution_match}")
        
        return shape_match and affine_match and resolution_match
    
    def register_mask_to_pet(self):
        """
        使用您提供的MATLAB代码方法将参考脑区mask配准到PET图像空间
        """
        print("开始图像配准...")
        
        # 保存临时文件用于配准
        temp_pet_path = "temp_pet.nii"
        temp_mask_path = "temp_mask.nii"
        
        try:
            # 保存临时文件
            nib.save(self.pet_img, temp_pet_path)
            nib.save(self.ref_mask_img, temp_mask_path)
            
            # 使用配准函数
            registered_mask_data, pet_affine = deform_img_based_on_other_img(
                temp_mask_path, temp_pet_path
            )
            
            # 创建新的mask图像
            self.registered_mask = nib.Nifti1Image(
                registered_mask_data.astype(np.uint8), 
                pet_affine, 
                self.pet_img.header
            )
            
            print("配准完成")
            return True
            
        except Exception as e:
            print(f"配准错误: {e}")
            return False
        
        finally:
            # 清理临时文件
            if os.path.exists(temp_pet_path):
                os.remove(temp_pet_path)
            if os.path.exists(temp_mask_path):
                os.remove(temp_mask_path)
    
    def calculate_suvr(self, use_registered_mask=True):
        """计算SUVR归一化图像"""
        print("计算SUVR...")
        
        # 选择使用的mask
        if use_registered_mask and self.registered_mask is not None:
            mask_data = self.registered_mask.get_fdata()
        else:
            mask_data = self.ref_mask_data
        
        # 确保mask与PET图像尺寸一致
        if mask_data.shape != self.pet_data.shape:
            print("警告: mask与PET图像尺寸不一致，使用配准后的mask")
            if self.registered_mask is not None:
                mask_data = self.registered_mask.get_fdata()
            else:
                raise ValueError("mask与PET图像尺寸不一致且无配准后的mask")
        
        # 提取参考脑区（mask值为1的区域）
        reference_region = self.pet_data[mask_data == 1]
        
        if len(reference_region) == 0:
            raise ValueError("参考脑区中没有有效的体素，请检查mask文件")
        
        # 计算参考脑区的平均强度
        reference_mean = np.mean(reference_region)
        print(f"参考脑区平均强度: {reference_mean:.4f}")
        print(f"参考脑区体素数量: {len(reference_region)}")
        
        # 计算SUVR
        suvr_data = self.pet_data / reference_mean
        
        # 创建新的NIfTI图像
        suvr_img = nib.Nifti1Image(suvr_data, self.pet_img.affine, self.pet_img.header)
        
        print(f"SUVR计算完成，范围: [{suvr_data.min():.4f}, {suvr_data.max():.4f}]")
        
        return suvr_img, suvr_data
    
    def save_suvr_image(self, suvr_img, output_path):
        """保存SUVR图像"""
        nib.save(suvr_img, output_path)
        print(f"SUVR图像已保存至: {output_path}")


# SUVR Calculate
def deform_img_based_on_other_img(original_img_path, refer_img_path, interpolation_order=0):
    """
    基于参考图像对原始图像进行空间变换
    """
    # 加载图像
    refer_img = nib.load(refer_img_path)
    original_img = nib.load(original_img_path)
    
    # 获取图像数据
    refer_data = refer_img.get_fdata()
    original_data = original_img.get_fdata()
    
    # 获取仿射矩阵
    refer_affine = refer_img.affine
    original_affine = original_img.affine
    
    # 获取图像尺寸
    refer_x, refer_y, refer_z = refer_data.shape
    origin_x, origin_y, origin_z = original_data.shape
    
    # 初始化新图像
    new_img = np.zeros((refer_x, refer_y, refer_z))
    
    # 计算从参考图像空间到原始图像空间的变换矩阵
    transform_matrix = np.linalg.inv(original_affine) @ refer_affine
    
    # 创建坐标网格
    i, j, k = np.meshgrid(
        np.arange(refer_x), 
        np.arange(refer_y), 
        np.arange(refer_z),
        indexing='ij'
    )
    
    # 将坐标展平以便批量处理
    coords = np.stack([i.flatten(), j.flatten(), k.flatten(), np.ones(i.size)]).T
    
    # 应用变换矩阵
    physical_coords = coords @ refer_affine.T
    origin_coords = physical_coords @ np.linalg.inv(original_affine).T
    
    # 提取坐标并重塑
    origin_i = origin_coords[:, 0].reshape(refer_x, refer_y, refer_z)
    origin_j = origin_coords[:, 1].reshape(refer_x, refer_y, refer_z)
    origin_k = origin_coords[:, 2].reshape(refer_x, refer_y, refer_z)
    
    # 使用插值获取原始图像值
    new_img = map_coordinates(
        original_data, 
        [origin_i, origin_j, origin_k], 
        order=interpolation_order,
        mode='constant', 
        cval=0.0
    )
    
    return new_img, refer_affine

def deform_img_based_on_other_img_exact(original_img_path, refer_img_path):
    """
    基于参考图像对原始图像进行空间变换
    """
    # 加载图像
    refer_img = nib.load(refer_img_path)
    original_img = nib.load(original_img_path)
    
    # 获取图像数据
    refer_data = refer_img.get_fdata()
    original_data = original_img.get_fdata()
    
    # 获取仿射矩阵
    refer_affine = refer_img.affine
    original_affine = original_img.affine
    
    # 获取图像尺寸
    refer_x, refer_y, refer_z = refer_data.shape
    origin_x, origin_y, origin_z = original_data.shape
    
    # 初始化新图像
    new_img = np.zeros((refer_x, refer_y, refer_z))
    
    # 逐个体素计算
    for i in range(refer_x):
        for j in range(refer_y):
            for k in range(refer_z):
                # 物理坐标转换
                physical_coor = np.array([i, j, k, 1]) @ refer_affine.T
                
                # 转换到原始图像坐标空间
                origin_coor = physical_coor @ np.linalg.inv(original_affine).T
                origin_coor = np.round(origin_coor[:3]).astype(int)
                
                # 边界检查
                origin_x_coord = max(min(origin_coor[0], origin_x - 1), 0)
                origin_y_coord = max(min(origin_coor[1], origin_y - 1), 0)
                origin_z_coord = max(min(origin_coor[2], origin_z - 1), 0)
                
                # 赋值
                new_img[i, j, k] = original_data[origin_x_coord, origin_y_coord, origin_z_coord]
    
    return new_img, refer_affine

def save_registered_image(registered_data, affine_matrix, output_path, dtype=None):
    """
    保存配准后的图像
    """
    if dtype is not None:
        registered_data = registered_data.astype(dtype)
    
    registered_img = nib.Nifti1Image(registered_data, affine_matrix)
    nib.save(registered_img, output_path)

def register_and_save(original_img_path, refer_img_path, output_path, 
                      method='vectorized', interpolation_order=0, dtype=None):
    """
    配准并保存图像的完整函数
    """
    # 选择配准方法
    if method == 'exact':
        registered_data, affine = deform_img_based_on_other_img_exact(
            original_img_path, refer_img_path
        )
    else:
        registered_data, affine = deform_img_based_on_other_img(
            original_img_path, refer_img_path, interpolation_order
        )
    
    # 保存配准后的图像
    save_registered_image(registered_data, affine, output_path, dtype)
    
    return registered_data, affine

def calculate_label_suvr(label_path, suv_path, labels):
    # 读取label文件
    label_img = nib.load(label_path)
    label_data = label_img.get_fdata()

    # 读取SUV文件
    suv_img = nib.load(suv_path)
    suv_data = suv_img.get_fdata()

    # 初始化一个字典来存储每个label的SUVR
    label_suvr = {}

    # 遍历每个label
    for label in labels:
        # 获取当前label的区域
        region_mask = label_data == label
        region_suv = suv_data[region_mask]

        # 计算当前label的SUV均值
        region_mean = np.nanmean(region_suv)

        # 计算SUVR
        suvr_value = region_mean
        label_suvr[f'Label{label}'] = suvr_value

    return label_suvr

def suvr_compute(label_path, pet_path, labels):
    label_path = os.path.join(label_path)
    pet_path = os.path.join(pet_path)

    # 计算每个label的SUVR
    label_suvr = calculate_label_suvr(label_path, pet_path, labels)

    return label_suvr

