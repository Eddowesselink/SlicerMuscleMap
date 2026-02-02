import os
import subprocess
import tempfile
import logging
import importlib.util
import shutil
import sys  # <-- NIEUW

import slicer
import vtk, qt, ctk
from slicer.ScriptedLoadableModule import *
import json
import urllib.request
import colorsys


MIN_PYTHON_VERSION = (3, 8)

LABELS_JSON_URL = (
    "https://raw.githubusercontent.com/MuscleMap/MuscleMap/main/scripts/models/wholebody/"
    "contrast_agnostic_wholebody_model.json"
)

class WholeBodyMuscleSegmentation(ScriptedLoadableModule):
    """Main entry point for the MuscleMap whole-body segmentation module."""

    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "MuscleMap Whole-body Segmentation"
        parent.categories = ["MuscleMap"]
        parent.contributors = ["Eddo Wesselink and Kenneth Arnold Weber"]

        # -----------------------------
        # HELP SECTION
        # -----------------------------
        parent.helpText = (
            "MuscleMap provides automated whole-body muscle segmentation directly inside 3D Slicer.\n\n"
            "Features:\n"
            " • Fully automated whole-body muscle segmentation\n"
            " • Support for CT and MRI\n"
            " • Export of muscle labels for quantitative research workflows\n\n"
            "MuscleMap requires the SlicerPyTorch extension. After installation, all dependencies\n"
            "can be installed using the 'Install MuscleMap dependencies' button.\n\n"
            "For documentation and source code, visit:\n"
            "https://github.com/MuscleMap/MuscleMap"
        )

        parent.acknowledgementText = (
            "MuscleMap is developed by the MuscleMap Consortium.\n\n"
            "If you use this extension in academic or scientific work, please cite the following publications:\n\n"
            "------------------------------------------------------------\n\n"
            "1) McKay MJ, Weber KA 2nd, Wesselink EO, Smith ZA, Abbott R, Anderson DB, Ashton-James CE, "
            "Atyeo J, Beach AJ, Burns J, Clarke S, Collins NJ, Coppieters MW, Cornwall J, Crawford RJ, "
            "De Martino E, Dunn AG, Eyles JP, Feng HJ, Fortin M, Franettovich Smith MM, Galloway G, "
            "Gandomkar Z, Glastras S, Henderson LA, Hides JA, Hiller CE, Hilmer SN, Hoggarth MA, Kim B, "
            "Lal N, LaPorta L, Magnussen JS, Maloney S, March L, Nackley AG, O'Leary SP, Peolsson A, "
            "Perraton Z, Pool-Goudzwaard AL, Schnitzler M, Seitz AL, Semciw AI, Sheard PW, Smith AC, "
            "Snodgrass SJ, Sullivan J, Tran V, Valentin S, Walton DM, Wishart LR, Elliott JM. "
            "MuscleMap: An Open-Source, Community-Supported Consortium for Whole-Body Quantitative MRI of Muscle. "
            "J Imaging. 2024;10(11):262. https://doi.org/10.3390/jimaging10110262\n\n"
            "------------------------------------------------------------\n\n"
            "2) Wesselink EO, Elliott JM, Coppieters MW, Hancock MJ, Cronin B, Pool-Goudzwaard A, "
            "Weber II KA. Convolutional neural networks for the automatic segmentation of lumbar "
            "paraspinal muscles in people with low back pain. Sci Rep. 2022;12(1):13485. "
            "https://doi.org/10.1038/s41598-022-16710-5\n\n"
            "------------------------------------------------------------\n\n"
            "MuscleMap is built upon open-source software including:\n"
            " • 3D Slicer (https://slicer.org)\n"
            " • MONAI (https://monai.io)\n"
            " • PyTorch (https://pytorch.org)\n"
            " • Nibabel\n\n"
            "We gratefully acknowledge the developers and contributors of these projects."
        )

        iconPath = os.path.join(
            os.path.dirname(__file__),
            "Resources", "Icons", "MuscleMap.png"
        )
        parent.icon = qt.QIcon(iconPath)

    def icon(self):
        iconPath = os.path.join(
            os.path.dirname(__file__),
            "Resources", "Icons", "MuscleMap.png"
        )
        return qt.QIcon(iconPath)


class WholeBodyMuscleSegmentationWidget(ScriptedLoadableModuleWidget):
    """User interface for the MuscleMap whole-body segmentation module."""

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        self.logic = WholeBodyMuscleSegmentationLogic()

        logoLabel = qt.QLabel()
        logoPath = os.path.join(os.path.dirname(__file__), "MuscleMap.png")
        pixmap = qt.QPixmap(logoPath)
        scaled = pixmap.scaledToWidth(200, qt.Qt.SmoothTransformation)
        logoLabel.setPixmap(scaled)
        logoLabel.setAlignment(qt.Qt.AlignCenter)
        self.layout.addWidget(logoLabel)

        packagesCollapsibleButton = ctk.ctkCollapsibleButton()
        packagesCollapsibleButton.text = "Installing packages"
        self.layout.addWidget(packagesCollapsibleButton)

        packagesLayout = qt.QVBoxLayout(packagesCollapsibleButton)

        self.extensionsButton = qt.QPushButton("Prerequisite: Complete this step before proceeding")
        self.extensionsButton.toolTip = (
            "Shows instructions for installing required python packages (e.g. PyTorch)."
        )
        self.extensionsButton.connect("clicked()", self.onExtensionsClicked)
        packagesLayout.addWidget(self.extensionsButton)

        self.installButton = qt.QPushButton("Install MuscleMap dependencies")
        self.installButton.toolTip = (
            "Install the MuscleMap toolbox into Slicer's Python using pip."
        )
        self.installButton.connect("clicked()", self.onInstallClicked)
        packagesLayout.addWidget(self.installButton)

        advancedCollapsibleButton = ctk.ctkCollapsibleButton()
        advancedCollapsibleButton.text = "Advanced"
        self.layout.addWidget(advancedCollapsibleButton)

        advancedLayout = qt.QFormLayout(advancedCollapsibleButton)

        self.forceCpuCheckBox = qt.QCheckBox("Force CPU (ignore GPU)")
        self.forceCpuCheckBox.setToolTip(
            "If checked, MuscleMap will run on the CPU even if a GPU is available."
        )
        self.forceCpuCheckBox.checked = False
        advancedLayout.addRow(self.forceCpuCheckBox)


        self.forceLowerOverlapCheckBox = qt.QCheckBox("Force lower overlap")
        self.forceLowerOverlapCheckBox.setToolTip(
            "If checked, run mm_segment with -s 75. Otherwise uses -s 90."
        )
        self.forceLowerOverlapCheckBox.checked = False
        advancedLayout.addRow(self.forceLowerOverlapCheckBox)

        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Inputs"
        self.layout.addWidget(parametersCollapsibleButton)

        parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

        self.loadFromFileButton = qt.QPushButton("Load volume from file...")
        self.loadFromFileButton.toolTip = "Select an image file from disk and load it as input volume."
        self.loadFromFileButton.connect("clicked()", self.onLoadFromFileClicked)
        parametersFormLayout.addRow(self.loadFromFileButton)

        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputSelector.selectNodeUponCreation = True
        self.inputSelector.addEnabled = False
        self.inputSelector.removeEnabled = False
        self.inputSelector.noneEnabled = True
        self.inputSelector.noneDisplay = ""  
        self.inputSelector.showHidden = False
        self.inputSelector.showChildNodeTypes = False
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.inputSelector.setToolTip("Select the input whole-body image.")
        parametersFormLayout.addRow("Input volume:", self.inputSelector)

        # Button: run segmentation
        self.runButton = qt.QPushButton("Run MuscleMap segmentation")
        self.runButton.toolTip = "Run mm_segment on the selected volume."
        self.runButton.connect("clicked()", self.onRunClicked)
        self.layout.addWidget(self.runButton)

        # Outputs
        outputsCollapsibleButton = ctk.ctkCollapsibleButton()
        outputsCollapsibleButton.text = "Outputs"
        self.layout.addWidget(outputsCollapsibleButton)

        outputsLayout = qt.QFormLayout(outputsCollapsibleButton)

        self.outputSegmentationSelector = slicer.qMRMLNodeComboBox()
        self.outputSegmentationSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.outputSegmentationSelector.selectNodeUponCreation = False
        self.outputSegmentationSelector.addEnabled = True
        self.outputSegmentationSelector.removeEnabled = True
        self.outputSegmentationSelector.noneEnabled = True
        self.outputSegmentationSelector.showHidden = False
        self.outputSegmentationSelector.showChildNodeTypes = False
        self.outputSegmentationSelector.setMRMLScene(slicer.mrmlScene)
        self.outputSegmentationSelector.setToolTip(
            "Select the segmentation for 3D visualization."
        )
        outputsLayout.addRow("Segmentation:", self.outputSegmentationSelector)

        self.show3DButton = slicer.qMRMLSegmentationShow3DButton()
        self.show3DButton.setToolTip("Show 3D representation of the selected segmentation.")
        outputsLayout.addRow(self.show3DButton)

        self.outputSegmentationSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)",
            self.show3DButton.setSegmentationNode
        )

        self.layout.addStretch(1)

    # --- UI callbacks ---

    def onLoadFromFileClicked(self):
        fileFilters = (
            "Volume files (*.nii *.nii.gz *.nrrd *.mha *.mhd);;"
            "All files (*)"
        )
        filePath = qt.QFileDialog.getOpenFileName(
            slicer.util.mainWindow(),
            "Select input volume",
            "",
            fileFilters
        )

        if not filePath:
            return

        volumeNode = slicer.util.loadVolume(filePath)
        if not volumeNode:
            slicer.util.errorDisplay(f"Failed to load volume from file:\n{filePath}")
            return

        self.inputSelector.setCurrentNode(volumeNode)

    def onExtensionsClicked(self):
        msg = (
            "MuscleMap requires the PyTorch package to be installed in 3D Slicer via the SlicerPyTorch extension.\n\n"
            "Step 1 – Install the 'PyTorch' (SlicerPyTorch) extension:\n"
            "  • In Slicer, go to:  View → Extensions Manager\n"
            "  • Search for:  PyTorch or SlicerPyTorch\n"
            "  • Install the extension and restart Slicer.\n\n"
            "Step 2 – Download and install PyTorch inside the SlicerPyTorch module:\n"
            "  • Go to:  View → Modules\n"
            "  • Choose:  Utilities → PyTorch (or 'SlicerPyTorch')\n"
            "  • Click the button 'Install PyTorch' and select the CPU or GPU build that matches your system.\n\n"
            "Step 3 – Return to the MuscleMap module and click 'Install MuscleMap dependencies'\n"
            "         to install the remaining Python packages used by MuscleMap.\n\n"
            "We recommend installing the appropriate PyTorch wheel using pip as suggested by the SlicerPyTorch module."
        )
        slicer.util.infoDisplay(msg, windowTitle="Prerequisite: Complete this step before proceeding")

    def onInstallClicked(self):
        progress = qt.QProgressDialog(
            "Installing MuscleMap dependencies...\n\n"
            "This may take a while, depending on your internet connection.",
            None,
            0,
            0,
            slicer.util.mainWindow()
        )
        progress.windowTitle = "Installing MuscleMap"
        progress.setWindowModality(qt.Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        slicer.app.processEvents()

        try:
            self.logic.ensureDependencies()
            progress.close()
            slicer.util.infoDisplay("MuscleMap dependencies are installed and ready.")
        except Exception as e:
            progress.close()
            slicer.util.errorDisplay(f"Failed to install/check dependencies:\n{e}")

    def onRunClicked(self):
        inputVolume = self.inputSelector.currentNode()
        if not inputVolume:
            slicer.util.errorDisplay("Please select an input volume first.")
            return

        # Read advanced option: force CPU
        force_cpu = bool(self.forceCpuCheckBox.isChecked()) if hasattr(self, "forceCpuCheckBox") else False
        force_lower_overlap = bool(self.forceLowerOverlapCheckBox.isChecked()) if hasattr(self, "forceLowerOverlapCheckBox") else False

        try:
            import torch
            if force_cpu:
                device = "CPU (forced)"
            else:
                if torch.cuda.is_available():
                    device = "GPU"
                else:
                    device = "CPU"
        except Exception:
            device = "CPU"

        msg = (
            "Running MuscleMap whole-body segmentation.\n"
            f"Computation device: {device}.\n"
        )

        if device.startswith("CPU"):
            msg += (
                "\nNote: Processing on CPU may be slow, depending on image size and system performance."
            )

        if force_lower_overlap:
            msg += (
                "\nOptional setting enabled: overlap is lowered (mm_segment -s 75)."
            )
        else:
            msg += (
                  "\nTip: For segmentation issues related to processing speed, enable 'Force lower overlap' (mm_segment -s 75)."
            )

        msg += (
            "\n\nIf you use MuscleMap models for research, "
            "please cite the MuscleMap publications (see module Acknowledgements)."
        )

        logging.info(f"[MuscleMap] {msg.replace(os.linesep, ' ')}")

        progress = qt.QProgressDialog(
            msg,
            None,
            0,
            0,
            slicer.util.mainWindow()
        )
        progress.setCancelButtonText("Cancel")
        progress.windowTitle = "MuscleMap segmentation"
        progress.setWindowModality(qt.Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        slicer.app.processEvents()

        try:
            segNode = self.logic.runSegmentation(inputVolume, force_cpu=force_cpu,force_lower_overlap=force_lower_overlap)
            if segNode:
                self.outputSegmentationSelector.setCurrentNode(segNode)
                self.show3DButton.setSegmentationNode(segNode)
        except Exception as e:
            slicer.util.errorDisplay(f"Segmentation failed:\n{e}")
        finally:
            progress.close()

class WholeBodyMuscleSegmentationLogic(ScriptedLoadableModuleLogic):
    """Processing code for the MuscleMap whole-body segmentation module."""

    def _fetch_labels_json(self, url: str) -> dict:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "3D Slicer MuscleMap"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _build_label_dataframe_and_color_node(self):
        """
        - Get labels from json from github
        - Build a dataframe 
        - Update  vtkMRMLColorTableNode where index == label value
        """
        data = self._fetch_labels_json(LABELS_JSON_URL)

        labels = data.get("labels", [])
        if not labels:
            raise RuntimeError("No 'labels' found in wholebody model json.")

        labels_sorted = sorted(labels, key=lambda x: int(x.get("value", 0)))
        values = [int(x["value"]) for x in labels_sorted]
        max_value = max(values)

        rows = []
        n = len(labels_sorted)

        sat = 0.75
        val = 0.90

        for i, item in enumerate(labels_sorted):
            region = (item.get("region") or "").strip()
            anatomy = (item.get("anatomy") or "").strip()
            side = (item.get("side") or "").strip()
            value_i = int(item["value"])

            if side and side.lower() != "no side":
                name = f"{anatomy} ({side})"
            else:
                name = f"{anatomy}"

            h = (i * 0.61803398875) % 1.0
            r, g, b = colorsys.hsv_to_rgb(h, sat, val)

            rows.append(
                {
                    "value": value_i,
                    "name": name,
                    "region": region,
                    "anatomy": anatomy,
                    "side": side,
                    "r": float(r),
                    "g": float(g),
                    "b": float(b),
                    "hex": "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255)),
                }
            )
        df = None
        try:
            import pandas as pd
            df = pd.DataFrame(rows, columns=["value", "name", "region", "anatomy", "side", "hex", "r", "g", "b"])
        except Exception:
            df = rows  # fallback

        colorNodeName = "MuscleMapWholeBodyLabels"
        existing = slicer.util.getNode(colorNodeName) if slicer.mrmlScene.GetFirstNodeByName(colorNodeName) else None

        if existing and existing.IsA("vtkMRMLColorTableNode"):
            colorNode = existing
        else:
            colorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode", colorNodeName)
            colorNode.SetTypeToUser()

        colorNode.SetNumberOfColors(max_value + 1)

        for idx in range(max_value + 1):
            colorNode.SetColor(idx, "", 0.0, 0.0, 0.0, 0.0)

        colorNode.SetColor(0, "background", 0.0, 0.0, 0.0, 0.0)

        for row in rows:
            colorNode.SetColor(int(row["value"]), row["name"], row["r"], row["g"], row["b"], 1.0)

        self._label_df = df
        self._label_color_node = colorNode
        return df, colorNode

    def ensureDependencies(self):
        if sys.version_info < MIN_PYTHON_VERSION:
            msg = (
                "MuscleMap requires at least Python "
                f"{MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}.\n\n"
                f"You are currently running Python {sys.version_info.major}.{sys.version_info.minor}.\n\n"
                "Please install a more recent 3D Slicer version "
                "(for example Slicer 5.2 or newer) from https://slicer.org "
                "and then reinstall the MuscleMap extension."
            )
            logging.error("[MuscleMap] " + msg.replace("\n", " "))
            try:
                slicer.util.errorDisplay(msg, windowTitle="MuscleMap – incompatible Python")
            except Exception:
                pass
            raise RuntimeError(msg)

        def have_module(name: str) -> bool:
            import importlib.util
            return importlib.util.find_spec(name) is not None

        have_monai       = have_module("monai")
        have_nibabel     = have_module("nibabel")
        have_portalocker = have_module("portalocker")
        have_pandas      = have_module("pandas")
        have_torch       = have_module("torch")
        have_sklearn     = have_module("sklearn")
        have_tqdm        = have_module("tqdm")
        have_joblib      = have_module("joblib")
        have_threadpool  = have_module("threadpoolctl")
        have_scipy       = have_module("scipy")
        have_pytz        = have_module("pytz")
        have_dateutil    = have_module("dateutil")

        if not have_torch:
            logging.error(
                "[MuscleMap] PyTorch (torch) is not accessible in Slicer."
            )
            raise RuntimeError(
                "PyTorch (torch) is not installed in this 3D Slicer Python environment.\n\n"
                "To install PyTorch correctly, please follow these steps:\n\n"
                "  1) Install the 'PyTorch' / 'SlicerPyTorch' extension:\n"
                "       • In Slicer, go to:  View → Extensions Manager\n"
                "       • Search for:  PyTorch or SlicerPyTorch\n"
                "       • Install the extension and restart Slicer\n\n"
                "  2) After restarting, open the SlicerPyTorch utility module:\n"
                "       • Go to:  View → Modules\n"
                "       • Choose:  Utilities → PyTorch (or 'SlicerPyTorch')\n\n"
                "  3) In that module, click the button:\n"
                "       “Install PyTorch”  (CPU or GPU version depending on your system)\n\n"
                "After completing these steps, return to MuscleMap and try again."
            )

        import torch
        import shutil
        have_mm_segment = shutil.which("mm_segment") is not None

        if all([
            have_monai, have_nibabel, have_torch, have_joblib, have_dateutil,
            have_pytz, have_threadpool, have_scipy, have_tqdm,
            have_portalocker, have_sklearn, have_pandas, have_mm_segment
        ]):
            logging.info("[MuscleMap] All dependencies already present, nothing to install.")
            return

        logging.info("[MuscleMap] Missing dependencies detected, installing...")

        minimal_packages = [
            ("monai",         "monai==1.5.1"),
            ("nibabel",       "nibabel==5.2.1"),
            ("tqdm",          "tqdm==4.67.1"),
            ("portalocker",   "portalocker==3.1.1"),
            ("pandas",        "pandas"),
            ("sklearn",       "scikit-learn"),
            ("joblib",        "joblib"),
            ("threadpoolctl", "threadpoolctl"),
            ("scipy",         "scipy"),
            ("pytz",          "pytz"),
            ("dateutil",      "python-dateutil"),
        ]

        for module_name, pkg in minimal_packages:
            if not have_module(module_name):
                logging.info(f"[MuscleMap] Installing (no-deps): {pkg}")
                slicer.util.pip_install(["--no-deps", pkg])
            else:
                logging.info(f"[MuscleMap] {pkg} already installed, skipping.")

        if not have_mm_segment:
            logging.info("[MuscleMap] Installing MuscleMap (no deps)...")
            slicer.util.pip_install(
                ["--no-deps", "git+https://github.com/MuscleMap/MuscleMap.git"]
            )
        else:
            logging.info("[MuscleMap] mm_segment already found, skipping MuscleMap install.")

        logging.info("[MuscleMap] Dependency check/installation finished.")

    def runSegmentation(self, inputVolumeNode, force_cpu: bool = False, force_lower_overlap: bool = False):
        """
        1) Export the selected Slicer volume to a temporary NIfTI file.
        2) Run 'mm_segment -i <input>' (provided by the MuscleMap toolbox).
        3) Detect the new NIfTI output file from mm_segment and load it as labelmap.

        """
        if not inputVolumeNode:
            raise ValueError("No input volume node provided.")

        self.ensureDependencies()

        tempDir = tempfile.mkdtemp(prefix="MuscleMap_")
        inputPath = os.path.join(tempDir, "input.nii.gz")

        logging.info(f"[MuscleMap] Saving input volume to: {inputPath}")
        if not slicer.util.saveNode(inputVolumeNode, inputPath):
            raise RuntimeError(f"Failed to save input volume to {inputPath}")

        before_files = {
            f for f in os.listdir(tempDir)
            if f.lower().endswith((".nii", ".nii.gz"))
        }

        s_value = "75" if force_lower_overlap else "90"
        cmd = ["mm_segment", "-i", inputPath, "-s", s_value]

        if force_cpu:
            cmd.extend(["-g", "N"])
        else:
            cmd.extend(["-g", "Y"])

        logging.info("[MuscleMap] Running command: " + " ".join(cmd))

        env = os.environ.copy()
        if force_cpu:
            env["CUDA_VISIBLE_DEVICES"] = ""
            logging.info("[MuscleMap] Forcing CPU execution (CUDA_VISIBLE_DEVICES='').")

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,     
            shell=False,
            cwd=tempDir,
            env=env,
        )
        logging.info("[MuscleMap] mm_segment stdout:\n" + result.stdout)
        logging.info("[MuscleMap] mm_segment stderr:\n" + result.stderr)

        if result.returncode != 0:
            raise RuntimeError(
                "mm_segment failed with non-zero exit code.\n\n"
            )

        after_files = {
            f for f in os.listdir(tempDir)
            if f.lower().endswith((".nii", ".nii.gz"))
        }

        new_files = sorted(list(after_files - before_files))

        if not new_files:
            raise RuntimeError(
                "No new NIfTI output found in "
                f"{tempDir} after running mm_segment.\n\n"
            )

        preferred = [
            f for f in new_files
            if any(key in f.lower() for key in ("dseg", "seg", "label"))
        ]
        outputFileName = preferred[0] if preferred else new_files[0]
        outputPath = os.path.join(tempDir, outputFileName)

        logging.info(f"[MuscleMap] Using output file: {outputPath}")

        labelNode = slicer.util.loadLabelVolume(outputPath)
        if not labelNode:
            raise RuntimeError("Failed to load the MuscleMap output labelmap.")
    
        # --- NEW: apply MuscleMap label names + colors from JSON (fixes VTK out-of-range colors) ---
        try:
            df, colorNode = self._build_label_dataframe_and_color_node()

            displayNode = labelNode.GetDisplayNode()
            if displayNode:
                displayNode.SetAndObserveColorNodeID(colorNode.GetID())
                displayNode.SetInterpolate(False)

            logging.info(f"[MuscleMap] Loaded {len(df) if hasattr(df, '__len__') else 'N/A'} label definitions from JSON.")
        except Exception as e:
            logging.warning(f"[MuscleMap] Could not apply JSON-based label colors/names: {e}")

        segmentationNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode", "MuscleMapSegmentation"
        )
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
            labelNode, segmentationNode
        )

        segmentationNode.CreateDefaultDisplayNodes()
        segmentationNode.GetDisplayNode().SetVisibility(True)

        slicer.mrmlScene.RemoveNode(labelNode)

        slicer.util.setSliceViewerLayers(background=inputVolumeNode)

        slicer.util.infoDisplay("MuscleMap whole-body segmentation completed.")

        return segmentationNode
