-include ../../config.mk

INCLUDES = -I ../../include -I ../../src -I ../../third_party

test_trace_to_idxs: test_trace_to_idxs.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test_trace_to_idxs test_trace_to_idxs.cpp $(LIBS) $(LDFLAGS)
	./test_trace_to_idxs
	rm test_trace_to_idxs

test_line_seg_tri_intersect: test_line_seg_tri_intersect.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test_line_seg_tri_intersect test_line_seg_tri_intersect.cpp $(LIBS) $(LDFLAGS)
	./test_line_seg_tri_intersect
	rm test_line_seg_tri_intersect

test_line_seg_box_overlap: test_line_seg_box_overlap.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test_line_seg_box_overlap test_line_seg_box_overlap.cpp $(LIBS) $(LDFLAGS)
	./test_line_seg_box_overlap
	rm test_line_seg_box_overlap

test_box_tri_overlap: test_box_tri_overlap.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test_box_tri_overlap test_box_tri_overlap.cpp $(LIBS) $(LDFLAGS)
	./test_box_tri_overlap
	rm test_box_tri_overlap

test_device_geometry: test_device_geometry.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test_device_geometry test_device_geometry.cpp $(LIBS) $(LDFLAGS)
	./test_device_geometry
	rm test_device_geometry

test_triangle_mesh_trigger: test_triangle_mesh_trigger.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test_triangle_mesh_trigger test_triangle_mesh_trigger.cpp $(LIBS) $(LDFLAGS)
	./test_triangle_mesh_trigger
	rm test_triangle_mesh_trigger
	python3 plot_normals.py

test_stl_mesh: test_stl_mesh.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test_stl_mesh test_stl_mesh.cpp $(LIBS) $(LDFLAGS)
	./test_stl_mesh
	rm test_stl_mesh
	python3 plot_normals.py

test_multi_mesh_id: test_multi_mesh_id.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test_multi_mesh_id test_multi_mesh_id.cpp $(LIBS) $(LDFLAGS)
	./test_multi_mesh_id
	rm test_multi_mesh_id
	python3 plot_multi_mesh.py
	rm *.txt

test_mesh_generator: test_mesh_generator.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test_mesh_generator test_mesh_generator.cpp $(LIBS) $(LDFLAGS)
	./test_mesh_generator
	rm test_mesh_generator
	python3 plot_multi_mesh.py
	rm *.txt

test_boundary_router: test_boundary_router.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test_boundary_router test_boundary_router.cpp $(LIBS) $(LDFLAGS)
	./test_boundary_router
	rm test_boundary_router
	python3 plot_animation.py
	rm *.txt

test_ray_tracing: test_ray_tracing.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test_ray_tracing test_ray_tracing.cpp $(LIBS) $(LDFLAGS)
	./test_ray_tracing
	python3 plt_to_png.py ray_tracing_img.plt
	rm test_ray_tracing

test_udf_ray_tracing: test_udf_ray_tracing.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test_udf_ray_tracing test_udf_ray_tracing.cpp $(LIBS) $(LDFLAGS)
	./test_udf_ray_tracing
	rm test_udf_ray_tracing

test_udf: test_udf.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test_udf test_udf.cpp $(LIBS) $(LDFLAGS)
	./test_udf
	rm test_udf