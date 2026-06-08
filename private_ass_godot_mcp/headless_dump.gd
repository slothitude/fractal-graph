extends SceneTree
## GAT Headless ClassDB Dump — runs with `godot --headless --script headless_dump.gd`.
## Outputs data/engine_schema.json and exits.

const OUTPUT_PATH := "data/engine_schema.json"
const IGNORED_CLASSES := [
	"GlobalConstants",
	"GDScript",
	"GDScriptNativeClass",
	"JavaClass",
	"JavaClassWrapper",
	"JNISingleton",
	"JSONParseResult",
	"Object",
	"_ArrayImmutableWrapper",
	"_ClassDB",
	"_EditorScript",
	"_File",
]


func _init() -> void:
	print("[GAT] Starting headless ClassDB dump...")

	var classes_array: Array = []
	var all_classes: PackedStringArray = ClassDB.get_class_list()
	var total: int = all_classes.size()
	var count: int = 0

	# Sort for deterministic output
	var sorted_classes: PackedStringArray = all_classes.duplicate()
	sorted_classes.sort()

	for cls_name in sorted_classes:
		if cls_name in IGNORED_CLASSES:
			continue
		if cls_name.begins_with("@"):
			continue
		if cls_name.begins_with("_") and not cls_name.begins_with("__"):
			continue

		var class_data := _extract_class(cls_name)
		classes_array.append(class_data)
		count += 1
		if count % 100 == 0:
			print("[GAT] %d/%d classes processed..." % [count, total])

	var schema := {
		"version": "4.6",
		"timestamp": Time.get_datetime_string_from_system(),
		"total_classes": classes_array.size(),
		"classes": classes_array,
	}

	# Ensure output directory exists
	var dir := DirAccess.open("res://")
	if dir and not dir.dir_exists("data"):
		dir.make_dir("data")

	var json_string := JSON.stringify(schema, "\t")
	var file := FileAccess.open(OUTPUT_PATH, FileAccess.WRITE)
	if file:
		file.store_string(json_string)
		file.close()
		print("[GAT] Dump complete: %d classes -> %s" % [classes_array.size(), OUTPUT_PATH])
	else:
		push_error("[GAT] Failed to write to %s" % OUTPUT_PATH)

	quit()


func _extract_class(cname: String) -> Dictionary:
	var parent: String = ClassDB.get_parent_class(cname)

	return {
		"class": cname,
		"inherits": parent if parent != "" else null,
		"is_abstract": not ClassDB.can_instantiate(cname),
		"is_singleton": Engine.has_singleton(cname),
		"properties": _extract_properties(cname),
		"methods": _extract_methods(cname),
		"signals": _extract_signals(cname),
		"constants": _extract_constants(cname),
		"enums": _extract_enums(cname),
	}


func _extract_properties(cname: String) -> Array:
	var result: Array = []
	var list: Array = ClassDB.class_get_property_list(cname)

	for prop in list:
		if prop["name"] == "script" or prop["name"] == "resource_name":
			continue
		result.append({
			"name": prop["name"],
			"type": _type_int_to_name(prop["type"]),
			"hint": _hint_int_to_name(prop["hint"]),
			"hint_string": prop.get("hint_string", ""),
			"usage": _usage_flags_to_list(prop["usage"]),
		})

	return result


func _extract_methods(cname: String) -> Array:
	var result: Array = []
	var list: Array = ClassDB.class_get_method_list(cname, true)

	# Only keep methods defined on this class, not inherited
	var parent: String = ClassDB.get_parent_class(cname)
	var parent_method_names: PackedStringArray = []
	if parent != "":
		var parent_list: Array = ClassDB.class_get_method_list(parent, true)
		for m in parent_list:
			parent_method_names.append(m["name"])

	for method in list:
		if method["name"] in parent_method_names:
			continue
		var args: Array = []
		if method.has("arguments"):
			for arg in method["arguments"]:
				args.append({
					"name": arg.get("name", ""),
					"type": _type_int_to_name(arg["type"]),
				})

		result.append({
			"name": method["name"],
			"return_type": _type_int_to_name(method.get("return_value", {}).get("type", 0)),
			"args": args,
			"is_virtual": method.get("flags", 0) & METHOD_FLAG_VIRTUAL != 0,
			"is_vararg": method.get("flags", 0) & METHOD_FLAG_VARARG != 0,
		})

	return result


func _extract_signals(cname: String) -> Array:
	var result: Array = []
	var list: Array = ClassDB.class_get_signal_list(cname, true)

	var parent: String = ClassDB.get_parent_class(cname)
	var parent_signal_names: PackedStringArray = []
	if parent != "":
		var parent_list: Array = ClassDB.class_get_signal_list(parent, true)
		for s in parent_list:
			parent_signal_names.append(s["name"])

	for sig in list:
		if sig["name"] in parent_signal_names:
			continue
		var args: Array = []
		if sig.has("arguments"):
			for arg in sig["arguments"]:
				args.append({
					"name": arg.get("name", ""),
					"type": _type_int_to_name(arg["type"]),
				})

		result.append({
			"name": sig["name"],
			"args": args,
		})

	return result


func _extract_constants(cname: String) -> Array:
	var result: Array = []
	var list: PackedStringArray = ClassDB.class_get_integer_constant_list(cname, true)

	var parent: String = ClassDB.get_parent_class(cname)
	var parent_constants: PackedStringArray = []
	if parent != "":
		parent_constants = ClassDB.class_get_integer_constant_list(parent, true)

	for const_name in list:
		if const_name in parent_constants:
			continue
		var value = ClassDB.class_get_integer_constant(cname, const_name)
		result.append({
			"name": const_name,
			"value": value,
		})

	return result


func _extract_enums(cname: String) -> Array:
	var result: Array = []
	var list: PackedStringArray = ClassDB.class_get_enum_list(cname, true)

	var parent: String = ClassDB.get_parent_class(cname)
	var parent_enums: PackedStringArray = []
	if parent != "":
		parent_enums = ClassDB.class_get_enum_list(parent, true)

	for enum_name in list:
		if enum_name in parent_enums:
			continue
		var values: Array = []
		var value_list: PackedStringArray = ClassDB.class_get_enum_constants(cname, enum_name, true)
		for v in value_list:
			values.append({
				"name": v,
				"value": ClassDB.class_get_integer_constant(cname, v),
			})
		result.append({
			"name": enum_name,
			"values": values,
		})

	return result


func _type_int_to_name(type_int: int) -> String:
	var types := {
		0: "null", 1: "bool", 2: "int", 3: "float", 4: "String",
		5: "Vector2", 6: "Vector2i", 7: "Rect2", 8: "Rect2i",
		9: "Vector3", 10: "Vector3i", 11: "Vector4", 12: "Vector4i",
		13: "Color", 14: "Plane", 15: "Quaternion", 16: "AABB",
		17: "Basis", 18: "Transform3D", 19: "Transform2D",
		20: "Projection", 21: "RID", 22: "Object", 23: "NodePath",
		24: "StringName", 25: "Dictionary", 26: "Array",
		27: "PackedByteArray", 28: "PackedInt32Array", 29: "PackedInt64Array",
		30: "PackedFloat32Array", 31: "PackedFloat64Array",
		32: "PackedStringArray", 33: "PackedVector2Array", 34: "PackedVector3Array",
		35: "PackedColorArray", 36: "PackedVector4Array",
		37: "Callable", 38: "Signal",
	}
	if types.has(type_int):
		return types[type_int]
	return str(type_int)


func _hint_int_to_name(hint_int: int) -> String:
	var hints := {
		0: "PROPERTY_HINT_NONE",
		1: "PROPERTY_HINT_RANGE",
		2: "PROPERTY_HINT_EXP_RANGE",
		3: "PROPERTY_HINT_LINK",
		4: "PROPERTY_HINT_FLAGS",
		5: "PROPERTY_HINT_LAYERS_2D_RENDER",
		6: "PROPERTY_HINT_LAYERS_2D_PHYSICS",
		7: "PROPERTY_HINT_LAYERS_3D_RENDER",
		8: "PROPERTY_HINT_LAYERS_3D_PHYSICS",
		9: "PROPERTY_HINT_FILE",
		10: "PROPERTY_HINT_DIR",
		11: "PROPERTY_HINT_GLOBAL_FILE",
		12: "PROPERTY_HINT_GLOBAL_DIR",
		13: "PROPERTY_HINT_RESOURCE_TYPE",
		14: "PROPERTY_HINT_MULTILINE_TEXT",
		15: "PROPERTY_HINT_EXPRESSION",
		16: "PROPERTY_HINT_PLACEHOLDER_TEXT",
		17: "PROPERTY_HINT_COLOR_NO_ALPHA",
		18: "PROPERTY_HINT_OBJECT_ID",
		19: "PROPERTY_HINT_TYPE_STRING",
		20: "PROPERTY_HINT_NODE_PATH",
		21: "PROPERTY_HINT_ENUM",
		22: "PROPERTY_HINT_ENUM_SUGGESTION",
		23: "PROPERTY_HINT_MAX",
		24: "PROPERTY_HINT_OBJECT_TOO_BIG",
		25: "PROPERTY_HINT_NODE_PATH_VALID_TYPES",
		26: "PROPERTY_HINT_SAVE_FILE",
		27: "PROPERTY_HINT_GLOBAL_SAVE_FILE",
		28: "PROPERTY_HINT_INT_IS_OBJECTID",
		29: "PROPERTY_HINT_INT_IS_POINTER",
		30: "PROPERTY_HINT_ARRAY_TYPE",
		31: "PROPERTY_HINT_LOCALE_ID",
		32: "PROPERTY_HINT_LOCALIZABLE_STRING",
	}
	if hints.has(hint_int):
		return hints[hint_int]
	return "PROPERTY_HINT_NONE"


func _usage_flags_to_list(usage: int) -> Array:
	var result: PackedStringArray = []
	if usage & PROPERTY_USAGE_STORAGE != 0:
		result.append("STORAGE")
	if usage & PROPERTY_USAGE_EDITOR != 0:
		result.append("EDITOR")
	if usage & PROPERTY_USAGE_SCRIPT_VARIABLE != 0:
		result.append("SCRIPT_VARIABLE")
	if usage & PROPERTY_USAGE_READ_ONLY != 0:
		result.append("READ_ONLY")
	if usage & PROPERTY_USAGE_DEFAULT != 0:
		result.append("DEFAULT")
	return Array(result)
