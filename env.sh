#!/bin/bash

# Get the script directory and construct the .env file path.
SCRIPT_PATH="${BASH_SOURCE[0]}"
if [[ "$SCRIPT_PATH" == */* ]]; then
    SCRIPT_DIR="${SCRIPT_PATH%/*}"
else
    SCRIPT_DIR="."
fi
ENV_FILE="$SCRIPT_DIR/.env"

# Exit if the file does not exist.
if [[ ! -f "$ENV_FILE" ]]; then
    echo ".env file not found."
    return 0 2>/dev/null || exit 0
fi

# Associative array for temporarily stored variables (requires Bash 4+).
declare -A PARSED_VARS

trim_whitespace() {
    TRIMMED_VALUE="$1"
    TRIMMED_VALUE="${TRIMMED_VALUE#"${TRIMMED_VALUE%%[![:space:]]*}"}"
    TRIMMED_VALUE="${TRIMMED_VALUE%"${TRIMMED_VALUE##*[![:space:]]}"}"
}

expand_variables() {
    local remaining="$1"
    local prefix variable_name replacement
    EXPANDED_VALUE=""

    while [[ -n "$remaining" ]]; do
        if [[ "$remaining" =~ ^([^'$']*)\$\{([A-Za-z_][A-Za-z0-9_]*)\}(.*)$ ]]; then
            prefix="${BASH_REMATCH[1]}"
            variable_name="${BASH_REMATCH[2]}"
            remaining="${BASH_REMATCH[3]}"
        elif [[ "$remaining" =~ ^([^'$']*)\$([A-Za-z_][A-Za-z0-9_]*)(.*)$ ]]; then
            prefix="${BASH_REMATCH[1]}"
            variable_name="${BASH_REMATCH[2]}"
            remaining="${BASH_REMATCH[3]}"
        else
            EXPANDED_VALUE+="$remaining"
            break
        fi

        if [[ -v "PARSED_VARS[$variable_name]" ]]; then
            replacement="${PARSED_VARS[$variable_name]}"
        else
            replacement="${!variable_name-}"
        fi
        EXPANDED_VALUE+="${prefix}${replacement}"
    done
}

# Read each line.
while IFS= read -r line || [[ -n "$line" ]]; do
    # Trim leading and trailing whitespace.
    trim_whitespace "$line"
    trimmed_line="$TRIMMED_VALUE"

    # Ignore blank lines and comments.
    if [[ -z "$trimmed_line" ]] || [[ "$trimmed_line" == \#* ]]; then
        continue
    fi

    # Extract the key and value from key=value entries.
    if [[ "$trimmed_line" =~ ^([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=[[:space:]]*(.*)$ ]]; then
        key="${BASH_REMATCH[1]}"
        rest="${BASH_REMATCH[2]}"

        # Handle single-quoted values.
        if [[ "$rest" =~ ^\'(.*)\'[[:space:]]*(#.*)?$ ]]; then
            value="${BASH_REMATCH[1]}"
        # Handle double-quoted values with escape sequences.
        elif [[ "$rest" =~ ^\"(.*)\"[[:space:]]*(#.*)?$ ]]; then
            raw_value="${BASH_REMATCH[1]}"
            printf -v value '%b' "$raw_value"
        # For unquoted values, remove a trailing comment.
        else
            if [[ "$rest" =~ ^(.*)[[:space:]]+#.*$ ]]; then
                rest="${BASH_REMATCH[1]}"
            fi
            trim_whitespace "$rest"
            value="$TRIMMED_VALUE"
        fi

        # Expand ${VAR} or $VAR without executing commands from the configuration.
        expand_variables "$value"
        expanded_value="$EXPANDED_VALUE"

        # Export the value to the environment.
        export "$key=$expanded_value"

        # Store the value for later variable substitutions.
        PARSED_VARS["$key"]="$expanded_value"
    fi
done < "$ENV_FILE"
