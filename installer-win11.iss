; Inno Setup script. Build the program first (build.bat or build_win7.bat), then open this file in Inno Setup and Compile.
; Default: Windows 10/11 64-bit build. For the Windows 7 build compile from the command line with:
;   ISCC.exe /DBUILD=Win7 installer.iss
#ifndef BUILD
  #define BUILD "Win64"
#endif
#if BUILD == "Win7"
  #define SrcDir "dist\Abasky-IQ-Win7"
  #define ExeName "Abasky-IQ-Win7.exe"
  #define OutName "Abasky-IQ-Setup-Win7"
#else
  #define SrcDir "dist\Abasky-IQ"
  #define ExeName "Abasky-IQ.exe"
  #define OutName "Abasky-IQ-Setup"
#endif

[Setup]
AppId={{7B1E0B0A-5C0D-4B65-9A2B-AB45C0A1D001}
AppName=Abasky-IQ
AppVersion=2.0
AppPublisher=Abasky
DefaultDirName={autopf}\Abasky-IQ
DefaultGroupName=Abasky-IQ
OutputDir=installer_out
OutputBaseFilename={#OutName}
SetupIconFile=iq.ico
UninstallDisplayIcon={app}\{#ExeName}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
#if BUILD != "Win7"
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
#endif

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#SrcDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\Abasky-IQ"; Filename: "{app}\{#ExeName}"
Name: "{autodesktop}\Abasky-IQ"; Filename: "{app}\{#ExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#ExeName}"; Description: "{cm:LaunchProgram,Abasky-IQ}"; Flags: nowait postinstall skipifsilent
