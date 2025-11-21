import os
import subprocess
import tempfile
import logging
import importlib.util
import shutil

import slicer
import vtk, qt, ctk
from slicer.ScriptedLoadableModule import *


#
# Module metadata
#

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

        # 👇 voeg dit toe
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

#
# GUI
#

class WholeBodyMuscleSegmentationWidget(ScriptedLoadableModuleWidget):
    """User interface for the MuscleMap whole-body segmentation module."""

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        self.logic = WholeBodyMuscleSegmentationLogic()
        
        logoLabel = qt.QLabel()
        logoPath = os.path.join(os.path.dirname(__file__), "MuscleMap.png")

        pixmap = qt.QPixmap(logoPath)

        # 👉 Schaal hier het logo (pas de breedte/hoogte naar wens aan!)
        scaled = pixmap.scaledToWidth(300, qt.Qt.SmoothTransformation)

        logoLabel.setPixmap(scaled)
        logoLabel.setAlignment(qt.Qt.AlignCenter)

        self.layout.addWidget(logoLabel)


        # Collapsible section
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "MuscleMap whole-body segmentation"
        self.layout.addWidget(parametersCollapsibleButton)

        parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

        # Input volume selector
        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputSelector.selectNodeUponCreation = True
        self.inputSelector.addEnabled = False
        self.inputSelector.removeEnabled = False
        self.inputSelector.noneEnabled = False
        self.inputSelector.showHidden = False
        self.inputSelector.showChildNodeTypes = False
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.inputSelector.setToolTip("Select the input whole-body image.")
        parametersFormLayout.addRow("Input volume:", self.inputSelector)

        # Button: install important Slicer extensions (e.g. PyTorch)
        self.extensionsButton = qt.QPushButton("Install important Slicer extensions")
        self.extensionsButton.toolTip = (
            "Shows instructions for installing required Slicer extensions (e.g. PyTorch)."
        )
        self.extensionsButton.connect("clicked()", self.onExtensionsClicked)
        self.layout.addWidget(self.extensionsButton)

        # Button: install / check dependencies
        self.installButton = qt.QPushButton("Install MuscleMap dependencies")
        self.installButton.toolTip = (
            "Install the MuscleMap toolbox into Slicer's Python using pip."
        )
        self.installButton.connect("clicked()", self.onInstallClicked)
        self.layout.addWidget(self.installButton)

        # Button: run segmentation
        self.runButton = qt.QPushButton("Run MuscleMap segmentation")
        self.runButton.toolTip = "Run mm_segment on the selected volume."
        self.runButton.connect("clicked()", self.onRunClicked)
        self.layout.addWidget(self.runButton)

        self.layout.addStretch(1)

    # --- UI callbacks ---

    def onExtensionsClicked(self):
        msg = (
            "MuscleMap requires the PyTorch extension to be installed in 3D Slicer.\n\n"
            "Please do the following steps once:\n"
            "  1. Open the Extensions Manager (puzzle-icon in the toolbar,\n"
            "     or via menu: View → Extensions Manager).\n"
            "  2. Search for 'PyTorch' (or 'SlicerPyTorch'). "
            "SlicerPyTorch automatically installs the best available PyTorch version for your system. "
            "If no compatible CUDA build is available, it will install a CPU-only version (this is normal behaviour).\n"
            "  3. Install that extension.\n"
            "  4. Close Slicer completely and start it again.\n\n"
            "After that, return to this module and click\n"
            "'Install MuscleMap dependencies' and then 'Run MuscleMap segmentation'."
        )

        slicer.util.infoDisplay(msg, windowTitle="Install important Slicer extensions")


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

        # Bepaal device (CPU / GPU)
        try:
            import torch
            if torch.cuda.is_available():
                device = "GPU"
            else:
                device = "CPU"
        except Exception:
            device = "CPU"

        # Message for user
        msg = (
            "Running MuscleMap whole-body segmentation.\n"
            f"Computation device: {device}.\n"
        )

        if device == "CPU":
            msg += (
                "\nNote: Processing on CPU may be slow, depending on image size and system performance."
            )

        msg += (
            "\n\nIf you use MuscleMap models for research, "
            "please cite the MuscleMap publications (see module Acknowledgements)."
        )

        logging.info(f"[MuscleMap] {msg.replace(os.linesep, ' ')}")

        # Progress dialoog tonen tijdens segmentatie
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
            self.logic.runSegmentation(inputVolume)
        except Exception as e:
            slicer.util.errorDisplay(f"Segmentation failed:\n{e}")
        finally:
            progress.close()

#
# Logic
#

class WholeBodyMuscleSegmentationLogic(ScriptedLoadableModuleLogic):
    """Processing code for the MuscleMap whole-body segmentation module."""

    def ensureDependencies(self):
        """
        Ensure that the minimal dependencies for mm_segment are available.
        Install only what is missing.
        """

        def have_module(name: str) -> bool:
            import importlib.util
            return importlib.util.find_spec(name) is not None

        # Check welke Python-packages we al hebben
        have_monai = have_module("monai")
        have_nibabel = have_module("nibabel")
        have_tqdm = have_module("tqdm")
        have_portalocker = have_module("portalocker")
        have_pandas = have_module("pandas")

        import shutil
        have_mm_segment = shutil.which("mm_segment") is not None

        if all([have_monai, have_nibabel, have_tqdm, have_portalocker, have_pandas, have_mm_segment]):
            logging.info("[MuscleMap] All dependencies already present, nothing to install.")
            return

        logging.info("[MuscleMap] Missing dependencies detected, installing...")

        # Let pip zelf de juiste pandas-versie kiezen (belangrijk voor Python 3.12)
        minimal_packages = [
            "monai==1.3.2",
            "nibabel==5.2.1",
            "tqdm==4.67.1",
            "portalocker==3.1.1",
            "pandas",  # geen versie-pin!
        ]

        for pkg in minimal_packages:
            module_name = pkg.split("==")[0]
            if not have_module(module_name):
                logging.info(f"[MuscleMap] Installing: {pkg}")
                slicer.util.pip_install(pkg)
            else:
                logging.info(f"[MuscleMap] {pkg} already installed, skipping.")

        if not have_mm_segment:
            logging.info("[MuscleMap] Installing MuscleMap (no deps)...")
            slicer.util.pip_install("git+https://github.com/MuscleMap/MuscleMap.git --no-deps")
        else:
            logging.info("[MuscleMap] mm_segment already found, skipping MuscleMap install.")

        logging.info("[MuscleMap] Dependency check/installation finished.")

    def runSegmentation(self, inputVolumeNode):
        """
        1) Export the selected Slicer volume to a temporary NIfTI file.
        2) Run 'mm_segment -i <input>' (provided by the MuscleMap toolbox).
        3) Detect the new NIfTI outputfile from mm_segment and load it as labelmap.
        """
        if not inputVolumeNode:
            raise ValueError("No input volume node provided.")

        # Make sure MuscleMap is available (installs if needed)
        self.ensureDependencies()

        # Temporary directory and file paths
        tempDir = tempfile.mkdtemp(prefix="MuscleMap_")
        inputPath = os.path.join(tempDir, "input.nii.gz")

        logging.info(f"[MuscleMap] Saving input volume to: {inputPath}")
        if not slicer.util.saveNode(inputVolumeNode, inputPath):
            raise RuntimeError(f"Failed to save input volume to {inputPath}")

        # Houd bij welke NIfTI's er al zijn vóór de run
        before_files = {
            f for f in os.listdir(tempDir)
            if f.lower().endswith(".nii.gz")
        }

        # Run mm_segment
        cmd = ["mm_segment", "-i", inputPath, "-s", "50"]
        logging.info("[MuscleMap] Running command: " + " ".join(cmd))

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            cwd=tempDir)
        logging.info("[MuscleMap] mm_segment stdout:\n" + result.stdout)
        logging.info("[MuscleMap] mm_segment stderr:\n" + result.stderr)

        if result.returncode != 0:
            raise RuntimeError("mm_segment failed, see the Python console for details or create an issue on https://github.com/MuscleMap/MuscleMap.")

        # Zoek nieuwe NIfTI-bestanden die mm_segment heeft aangemaakt
        after_files = {
            f for f in os.listdir(tempDir)
            if f.lower().endswith(".nii.gz")
        }
        new_files = sorted(list(after_files - before_files))

        if not new_files:
            raise RuntimeError(
                f"No new NIfTI output found in {tempDir} after running mm_segment."
            )

        # Als er meerdere zijn, kies er eentje met 'dseg'/'seg'/'label' in de naam als die bestaat
        preferred = [
            f for f in new_files
            if any(key in f.lower() for key in ("dseg", "seg", "label"))
        ]
        outputFileName = preferred[0] if preferred else new_files[0]
        outputPath = os.path.join(tempDir, outputFileName)

        logging.info(f"[MuscleMap] Using output file: {outputPath}")

        # 1) Laad als labelmap volume
        labelNode = slicer.util.loadLabelVolume(outputPath)
        if not labelNode:
            raise RuntimeError("Failed to load the MuscleMap output labelmap.")

        # 2) Maak een segmentation node en importeer de labelmap daarin
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode", "MuscleMapSegmentation"
        )
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
            labelNode, segmentationNode
        )

        # 3) Zorg dat de segmentatie zichtbaar is in de slice views
        segmentationNode.CreateDefaultDisplayNodes()
        segmentationNode.GetDisplayNode().SetVisibility(True)

        # (optioneel) oude labelmap node weggooien, is niet meer nodig
        slicer.mrmlScene.RemoveNode(labelNode)

        # 4) Achtergrond op je input; segmentatie wordt als overlay getoond
        slicer.util.setSliceViewerLayers(background=inputVolumeNode)

        slicer.util.infoDisplay("MuscleMap whole-body segmentation completed.")
