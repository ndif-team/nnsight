#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

// Structure to store mounted functions
typedef struct {
    PyObject* func;
    char* name;
} MountedFunction;

// Array to store mounted functions
static MountedFunction* mounted_functions = NULL;
static int num_mounted_functions = 0;
static int mounted_functions_capacity = 0;

static PyObject* get_dict(PyTypeObject* type) {
#if PY_VERSION_HEX < 0x030C0000
    PyObject* dict = type->tp_dict;
#else
    PyObject* dict = PyType_GetDict(type);
#endif
    return dict;
}

// Function to mount a new function
static PyObject* mount_function(PyObject* self, PyObject* args) {
    PyObject* func;
    const char* mount_point;
    
    if (!PyArg_ParseTuple(args, "Os", &func, &mount_point)) {
        return NULL;
    }
    
    if (!PyCallable_Check(func)) {
        PyErr_SetString(PyExc_TypeError, "First argument must be callable");
        return NULL;
    }
    
    // Check if we need to resize our array
    if (num_mounted_functions >= mounted_functions_capacity) {
        int new_capacity = mounted_functions_capacity == 0 ? 4 : mounted_functions_capacity * 2;
        MountedFunction* new_array = PyMem_Realloc(mounted_functions, 
                                                  new_capacity * sizeof(MountedFunction));
        if (!new_array) {
            PyErr_NoMemory();
            return NULL;
        }
        mounted_functions = new_array;
        mounted_functions_capacity = new_capacity;
    }
    
    // Store the function
    Py_INCREF(func);
    mounted_functions[num_mounted_functions].func = func;
    mounted_functions[num_mounted_functions].name = strdup(mount_point);
    num_mounted_functions++;

    PyObject* method = PyInstanceMethod_New(func);
    if (method) {
        PyObject *dict = get_dict(&PyBaseObject_Type);
        PyDict_SetItemString(dict, mount_point, method);
        Py_DECREF(method);
        PyType_Modified(&PyBaseObject_Type);
    }
    
    Py_RETURN_NONE;
}

// Function to unmount a function by name
static PyObject* unmount_function(PyObject* self, PyObject* args) {
    const char* mount_point;
    
    if (!PyArg_ParseTuple(args, "s", &mount_point)) {
        return NULL;
    }
    
    // Remove from base object type
    PyObject *dict = get_dict(&PyBaseObject_Type);
    PyDict_DelItemString(dict, mount_point);
    PyType_Modified(&PyBaseObject_Type);
    
    // Find and remove from our array
    for (int i = 0; i < num_mounted_functions; i++) {
        if (strcmp(mounted_functions[i].name, mount_point) == 0) {
            // Found the function, remove it
            Py_DECREF(mounted_functions[i].func);
            free(mounted_functions[i].name);
            
            // Shift remaining elements
            for (int j = i; j < num_mounted_functions - 1; j++) {
                mounted_functions[j] = mounted_functions[j + 1];
            }
            num_mounted_functions--;
            break;
        }
    }
    
    Py_RETURN_NONE;
}



// Method definitions
static PyMethodDef module_methods[] = {
    {"mount", mount_function, METH_VARARGS, "Mount a function to all objects"},
    {"unmount", unmount_function, METH_VARARGS, "Unmount a function from all objects"},
    {NULL, NULL, 0, NULL}  // Sentinel
};

// Module initialization function
PyMODINIT_FUNC PyInit_py_mount(void) {
    // Initialize mounted functions array
    mounted_functions = NULL;
    num_mounted_functions = 0;
    mounted_functions_capacity = 0;
    
    // Create and return the module
    static struct PyModuleDef def = {
        PyModuleDef_HEAD_INIT,
        "py_mount",
        "Adds methods to all Python objects",
        -1,
        module_methods
    };
    
    PyObject* module = PyModule_Create(&def);
    
    return module;
}