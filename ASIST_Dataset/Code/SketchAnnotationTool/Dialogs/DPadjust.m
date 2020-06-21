function varargout = DPadjust(varargin)
% DPADJUST MATLAB code for DPadjust.fig
%      DPADJUST, by itself, creates a new DPADJUST or raises the existing
%      singleton*.
%
%      H = DPADJUST returns the handle to a new DPADJUST or the handle to
%      the existing singleton*.
%
%      DPADJUST('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in DPADJUST.M with the given input arguments.
%
%      DPADJUST('Property','Value',...) creates a new DPADJUST or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before DPadjust_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to DPadjust_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help DPadjust

% Last Modified by GUIDE v2.5 10-Apr-2015 10:42:56

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @DPadjust_OpeningFcn, ...
                   'gui_OutputFcn',  @DPadjust_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before DPadjust is made visible.
function DPadjust_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to DPadjust (see VARARGIN)

% Choose default command line output for DPadjust
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes DPadjust wait for user response (see UIRESUME)
 uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = DPadjust_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
handles.output = 0;
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
% +1
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global adjustDecision;
adjustDecision = +1;
assignin('base','adjustDecision',adjustDecision);

% Use UIRESUME instead of delete because the OutputFcn needs
% to get the updated handles structure.
uiresume(handles.figure1);


% --- Executes on button press in pushbutton2.
% -1
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global adjustDecision;
adjustDecision = -1;
assignin('base','adjustDecision',adjustDecision);

% Use UIRESUME instead of delete because the OutputFcn needs
% to get the updated handles structure.
uiresume(handles.figure1);


% --- Executes on button press in pushbutton3.
% Done
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global adjustDecision;
adjustDecision = 0;
assignin('base','adjustDecision',adjustDecision);

% Use UIRESUME instead of delete because the OutputFcn needs
% to get the updated handles structure.
% The figure can be deleted now
delete(handles.figure1);


% --- Executes on button press in pushbutton4.
% +3
function pushbutton4_Callback(hObject, eventdata, handles) 
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global adjustDecision;
adjustDecision = +3;
assignin('base','adjustDecision',adjustDecision);

% Use UIRESUME instead of delete because the OutputFcn needs
% to get the updated handles structure.
uiresume(handles.figure1);


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global adjustDecision;
adjustDecision = +5;
assignin('base','adjustDecision',adjustDecision);

% Use UIRESUME instead of delete because the OutputFcn needs
% to get the updated handles structure.
uiresume(handles.figure1);


% --- Executes on button press in pushbutton6.
% -3
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global adjustDecision;
adjustDecision = -3;
assignin('base','adjustDecision',adjustDecision);

% Use UIRESUME instead of delete because the OutputFcn needs
% to get the updated handles structure.
uiresume(handles.figure1);


% --- Executes on button press in pushbutton7.
% -5
function pushbutton7_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global adjustDecision;
adjustDecision = -5;
assignin('base','adjustDecision',adjustDecision);

% Use UIRESUME instead of delete because the OutputFcn needs
% to get the updated handles structure.
uiresume(handles.figure1);
