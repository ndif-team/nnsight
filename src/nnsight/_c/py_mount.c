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
        // Inject into all currently loaded types
    // PyObject* modules = PyImport_GetModuleDict();
    // PyObject* key, *value;
    // Py_ssize_t pos = 0;
    
    // while (PyDict_Next(modules, &pos, &key, &value)) {
    //     if (!PyModule_Check(value)) continue;
        
    //     PyObject* module_dict = PyModule_GetDict(value);
    //     if (!module_dict) continue;
        
    //     PyObject* item_key, *item_value;
    //     Py_ssize_t item_pos = 0;
    //     while (PyDict_Next(module_dict, &item_pos, &item_key, &item_value)) {
    //         if (PyType_Check(item_value)) {
    //             PyTypeObject* type = (PyTypeObject*)item_value;
                
    //             // Create an instance method
    //             PyObject* method = PyInstanceMethod_New(func);
    //             if (method) {
    //                 PyDict_SetItemString(type->tp_dict, mount_point, method);
    //                 Py_DECREF(method);
    //                 PyType_Modified(type);
    //             }
    //         }
    //     }
    // Mount only on the base object type
    PyObject* method = PyInstanceMethod_New(func);
    if (method) {
        PyDict_SetItemString(PyBaseObject_Type.tp_dict, mount_point, method);
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
    PyDict_DelItemString(PyBaseObject_Type.tp_dict, mount_point);
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

// Cleanup function for mounted functions
static void cleanup_mounted_functions(void) {
    for (int i = 0; i < num_mounted_functions; i++) {
        Py_XDECREF(mounted_functions[i].func);
        free(mounted_functions[i].name);
    }
    PyMem_Free(mounted_functions);
    mounted_functions = NULL;
    num_mounted_functions = 0;
    mounted_functions_capacity = 0;
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
    if (module) {
        // Register cleanup function
        Py_AtExit(cleanup_mounted_functions);
    }
    
    return module;
}