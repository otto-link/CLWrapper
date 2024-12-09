## Macro-Logger

A simplified logging system using macros for C/C++ and Objective-C.

## Notes

#### Levels

There are three different logging levels. It's possible also disable all logs.

  - [0] NO_LOGS
  - [1] ERROR
  - [2] INFO
  - [3] DEBUG

The log level can be specified on compilation time using the macro 

```
LOG_LEVEL=level_number
```

If not specified is used the debug level.

#### Technical aspects

  * **Each log below the log level defined will not be included on binary.** This is relevant because allows save space and resources.
  * Logs are printed using fprintf function in conjunction with standard error stream (stderr) in C/C++ and Objective-C.
  * The log functions can handle arguments like printf or just a message. The message can be a C/C++ string or an Objective-C string.

	```C
	int x = 5;
	LOG_INFO("x * x = %d", x * x);
	LOG_DEBUG(@"Objective-C also supported");
	```
  * The log functions can receive Objective-C objects as arguments.

	```Objective-C
	NSString *baseStation = @"Houston";
	LOG_ERROR(@"%@ we have a problem!", baseStation);
	```

## Basic usage

The only file needed to use the macrologger is the header file, `macrologger.h`.

```C++

// C/C++
#include "macrologger.h"

// Objective-C
#import "macrologger.h"
```

### Available functions

```C++


LOG_DEBUG(message, arguments);

LOG_INFO(message, arguments);

LOG_ERROR(message, arguments);

LOG_IF_ERROR(condition, message, arguments);

```

### Examples

```C++


// C/C++ strings
LOG_DEBUG("Maybe i can touch this button...");
    
// Objective-C strings
LOG_INFO("@Pressure is dropping...");

// printing Objective-C objects    
NSString *baseStation = @"Houston";
LOG_ERROR(@"%@ we have a problem!", baseStation);

int going_down = 1;
LOG_IF_ERROR(going_down, "i'm going down... if only i had used macro-logger...");

```

### Sample output

```
2012-09-04 01:05:16 | DEBUG   | sample.m        | main:29 | Maybe i can touch this button...
2012-09-04 01:05:16 | INFO    | sample.m        | main:31 | Pressure is dropping...
2012-09-04 01:05:16 | ERROR   | sample.m        | main:34 | Houston we have a problem!
2012-09-04 01:05:16 | ERROR   | sample.m        | main:37 | i'm going down... if only i had used macro-logger...
```